import logging
from typing import Optional, Tuple, Dict, Any

from src.env import state
import copy
import json
from src.env.state import GreenState, GreenHistoryItem, get_active_subquery
from src.agent import actions, workers
from src.env.retriever import EphemeralRetriever

# Get the cost table
try:
    with open("data/meta/cost_table.json", "r") as f:
        COST_TABLE = json.load(f)
except FileNotFoundError:
    print("Warning: cost_table.json not found. Using default costs (1.0).")
    COST_TABLE = {}

trace_logger = logging.getLogger("LLM_TRACE")
trace_logger.addHandler(logging.NullHandler())

# A class to encapsulate the engine logic and manage the state
class GreenEngine:
    def __init__(self, retriever: EphemeralRetriever):
        self.retriever = retriever
    
    def get_cost(self, action_id: int) -> float:
        return COST_TABLE.get(str(action_id), 1.0)
    
    def step(self, state: GreenState, action_id: int, argument: Optional[str] = None) -> GreenState:
        """
        For our State Machine
        This function constitutes the universal transiation function:
        S' = T(S, A)
        
        Args:
            state (GreenState): Current state of the agent.
            action_id (int): The action to perform.
            argument (Optional[str]): Additional argument for the action.
        
        Returns:
            new_state (GreenState): The updated state after action execution.
        """
        logging.debug(f"Engine Step: Action ID {action_id} with argument: {argument}")
        # Deep copy the state to avoid mutating the original
        new_state = copy.deepcopy(state)

        # get_active_subquery is causing problems
        # trying new method where we get the active subquery index instead of the object itself, then we can update the state directly without worrying about references
        active_idx = -1
        active_subquery = None
        
        # Iterate backwards to match your stack logic (LIFO/Reversed)
        # using enumerate to keep track of the REAL index in the main list
        subqueries = new_state.get('subqueries', [])
        for i in range(len(subqueries) - 1, -1, -1):
            if subqueries[i]['status'] in ["ACTIVE", "PENDING"]:
                active_idx = i
                active_subquery = subqueries[i]
                # Side effect: Mark as ACTIVE immediately in the state
                new_state['subqueries'][i]['status'] = "ACTIVE" 
                break

        # Execute the action using the engine function
        # We don't use the obs or argument for generation
        obs = ""

        logging.debug(f"Attempting Action ID {action_id} on State with status: {new_state['status']}")

        # --- [0] or [1]: ANSWERING (GEN_SLM / GEN_LLM) ---
        if action_id in [actions.ACTION_GEN_SLM, actions.ACTION_GEN_LLM]:
            use_llm = (action_id == actions.ACTION_GEN_LLM)

            # Hold on, doesn't this depend on whether or not we have a subquery too? We need to know what to pass here.
            # This function will figure out what the active query is on its own
            answer = workers.generate_answer(new_state, use_llm=use_llm)

            trace_logger.debug(f"Do we have an active subquery? {'Yes' if get_active_subquery(new_state) is not None else 'No'}")
            trace_logger.debug(f"LLM RESPONDS: {answer}")

            # Check for active subquery using the canonical function (not stale local variable)
            current_active = get_active_subquery(new_state)
            
            # If we have an active subquery, we update that
            if current_active:
                # Find the index of this subquery so we can update it
                for i, sub in enumerate(new_state['subqueries']):
                    if sub['id'] == current_active['id']:
                        new_state['subqueries'][i]['answer'] = answer
                        new_state['subqueries'][i]['status'] = "ANSWERED"
                        break
                obs = f"Sub-query answered: {answer}"
                logging.debug(f"Updated active subquery with answer: {answer}")
                trace_logger.debug(f"Sub-query answered:\nQ: {current_active['question']}\nA: {answer}")
                trace_logger.debug(f"This is the state of all subqueries after answering:\n" + "\n".join([f"{sub['question']} - {sub['status']}" for sub in new_state['subqueries']]))
                
                new_subquery = get_active_subquery(new_state)
                new_q_text = new_subquery['question'] if new_subquery else "None"

                trace_logger.debug(f"If we look for active query now we get: {new_q_text}")

            else:
                # No active subquery - this is the main answer
                new_state['answer'] = answer
                new_state['status'] = "SOLVED"
                obs = f"Main query answered: {answer}"
                trace_logger.debug(f"Main query answered: {answer}")

        # --- [2] or [3]: RETRIEVAL (RET_KEY / RET_VEC) ---
        # This will do for now but I'd like the model to know whether the it's doing
        # a keyword or vector search
        elif action_id in [actions.ACTION_RET_KEY, actions.ACTION_RET_VEC]:
            # Execute Search
            # For now we use an SLM for keyword, LLM for vector
            # SLM is faster and cheaper for simple keyword generation
            # LLM is better at semantic understanding
            
            # If both retrievals use the SLM, the costs are:
            # RET_KEY: 127.5712 Joules (avg)
            # RET_VEC 189.2028 Joules (avg)
            # Accordingly we use SLM for keyword and LLM for vector to maintain the intuition of "keyword retrieval is cheaper but less powerful, vector retrieval is more expensive but more powerful"
            if action_id == actions.ACTION_RET_KEY:
                argument = workers.generate_query_for_keyword_search(new_state, use_llm=False)
                raw_docs = self.retriever.search_bm25(argument)
            else:
                argument = workers.generate_query_for_vector_search(new_state, use_llm=True)
                raw_docs = self.retriever.search_dense(argument)
            
            # Format & Update State
            formatted_docs = self._format_docs(raw_docs)
            if active_subquery is not None:
                active_subquery['documents'].extend(formatted_docs)
            else:
                new_state['documents'].extend(formatted_docs)
            
            obs = f"Found {len(formatted_docs)} docs."

        # [4] or [5]: GRADING (GRD_SLM / GRD_LLM)
        elif action_id in [actions.ACTION_GRD_SLM, actions.ACTION_GRD_LLM]:
        # Grade the documents in the active subquery
        # Not checked, may not work
            count_rel = 0
            target_docs = active_subquery['documents'] if active_subquery is not None else state.get('documents', [])

            if not target_docs:
                obs = "No documents to grade."
            else:
                logging.debug(f"Grading {len(target_docs)} documents for relevance.")
                use_llm = (action_id == actions.ACTION_GRD_LLM)
                
                for doc in target_docs:
                    grade = workers.generate_grade(new_state, doc["content"], use_llm=use_llm)
                    doc['relevance'] = "RELEVANT" if grade == "Relevant" else "IRRELEVANT"
                    if grade == "Relevant": count_rel += 1

                relevant_indices = []
                for i, doc in enumerate(target_docs):
                    if doc.get('relevance') == "RELEVANT":
                        relevant_indices.append(f"Doc {i+1} ({doc['title']})")
                        
                if relevant_indices:
                    obs = f"Graded docs. {count_rel} relevant: {', '.join(relevant_indices)}"
                else:
                    obs = "Graded docs. None found relevant."

       # [6]: REWRITE (RWT_SLM)
        elif action_id == actions.ACTION_RWT_SLM:           
            if active_subquery is not None:
                old_query = active_subquery.get("question", "")
                
                # Generate the new, specific question using past answers
                new_query = workers.generate_rewrite(new_state)
                
                if new_query and new_query.lower() != old_query.lower():
                    active_subquery["question"] = new_query
                    obs = f"Rewrote pending sub-query to: '{new_query}'"
                else:
                    obs = "Rewrite deemed unnecessary or failed. Kept original query."
            else:
                obs = "Action failed. No active sub-query to rewrite."

        # [7] or [8]: DECOMPOSITION (DEC_SLM / DEC_LLM)
        elif action_id in [actions.ACTION_DEC_SLM, actions.ACTION_DEC_LLM]:
            use_llm = (action_id == actions.ACTION_DEC_LLM)
            plan_text = workers.generate_plan(state, use_llm=use_llm)
            
            # Format the plan into subqueries
            # Not sure how well this is going to work but we can iterate

            # Previously this would recursively decompose the active subquery
            # I'm going to change the logic here
            # Now we always decompose the MAIN query, overwriting any existing subqueries
            lines = plan_text.split('\n')
            new_subs = []

            for i, line in enumerate(lines):
                clean = line.strip().lstrip('1234567890. ')
                if clean:
                    new_subs.append({
                        "id": f"{i}",
                        "question": clean,
                        "status": "PENDING",
                        "answer": None,
                        "documents": []
                    })

            new_state['subqueries'] = list(reversed(new_subs))            
            task_preview = "\n".join([f"{i}. {sub['question']}" for i, sub in enumerate(new_subs)])
            obs = f"Decomposed into {len(new_subs)} sub-tasks:\n{task_preview}"
                
        # [9]: FAILURE
        elif action_id == actions.ACTION_FAIL:
            obs = "Agent declared failure."
            new_state['status'] = "FAILED" # <--- Persist the failure

        # --- FALLBACK ---
        else:
            obs = "Invalid or No-Op Action."

        # History and Cost Update
        step_cost = self.get_cost(action_id)
        new_state['total_joules'] += step_cost

        new_state['history'].append(GreenHistoryItem(
            action_id=action_id,
            action_name=actions.get_action_name(action_id),
            observation=obs,
            argument=argument if argument is not None else "",
            cost=step_cost
        ))

        logging.debug(f"Before Step End: New State status: {new_state['status']}, action: {action_id}, observation: {obs}")
        return new_state
    
    # Helper functions    
    def _format_docs(self, raw_docs: list) -> list:
        # Logic to splite Title: Content into dicts
        # Update State
        formatted_docs = []
        for r in raw_docs:
            # Retriever returns strings "Title: Content"
            parts = r.split(": ", 1)
            title = parts[0] if len(parts)>1 else "Unknown"
            content = parts[1] if len(parts)>1 else r
            formatted_docs.append({"title": title, "content": content, "relevance": "UNKNOWN"})

        return formatted_docs
       
