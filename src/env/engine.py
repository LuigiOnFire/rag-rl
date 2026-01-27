import logging
from typing import Optional, Tuple, Dict, Any

from src.env import state
import copy
import json
from src.env.state import GreenState, GreenHistoryItem, get_active_subquery, is_main_query
from src.agent import actions, workers
from src.env.retriever import EphemeralRetriever

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.DEBUG)

# Get the cost table
try:
    with open("data/meta/cost_table.json", "r") as f:
        COST_TABLE = json.load(f)
except FileNotFoundError:
    print("Warning: cost_table.json not found. Using default costs (1.0).")
    COST_TABLE = {}

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

        # Identify which subquery that will be influenced by actions the operate on subqueries
        active_subquery = get_active_subquery(new_state)

        # Execute the action using the engine function
        obs = ""
        final_argument = argument or ""

        logging.debug(f"Attempting Action ID {action_id} on State with status: {new_state['status']}")

        # --- [0] or [1]: ANSWERING (GEN_SLM / GEN_LLM) ---
        # Currently this will answer the active SUBQUERY
        # not necessarily the GLOBAL question
        # I'm not sure I like this direction but we can change later
        if action_id in [actions.ACTION_GEN_SLM, actions.ACTION_GEN_LLM]:
            use_llm = (action_id == actions.ACTION_GEN_LLM)
            obs = workers.generate_answer(new_state, use_llm=use_llm)

            logging.debug(f"Do we have an active subquery? {'Yes' if active_subquery else 'No'}")
            logging.debug(f"LLM RESPONDS: {obs}")
            if active_subquery:
                # Check for Global Solved Status
                # If the active subquery is a GreenState instead of a SubQuery,
                # this object represents the main query
                # if it's solved, we mark the whole state as SOLVED
                if is_main_query(active_subquery):
                    logging.debug("Main query answered, marking state as SOLVED.")
                    new_state['status'] = "SOLVED"
        
        # --- [2] or [3]: RETRIEVAL (RET_KEY / RET_VEC) ---
        # This will do for now but I'd like the model to know whether the it's doing
        # a keyword or vector search
        elif action_id in [actions.ACTION_RET_KEY, actions.ACTION_RET_VEC]:
            # Execute Search
            # For now we use an SLM for keyword, LLM for vector
            # SLM is faster and cheaper for simple keyword generation
            # LLM is better at semantic understanding
            if action_id == actions.ACTION_RET_KEY:
                argument = workers.generate_query_for_keyword_search(new_state, use_llm=False)
                raw_docs = self.retriever.search_bm25(argument)
            else:
                argument = workers.generate_query_for_vector_search(new_state, use_llm=True)
                raw_docs = self.retriever.search_dense(argument)
            
            # Format & Update State
            formatted_docs = self._format_docs(raw_docs)
            if active_subquery:
                active_subquery['documents'].extend(formatted_docs)
            
            obs = f"Found {len(formatted_docs)} docs."

        # [4] or [5]: GRADING (GRD_SLM / GRD_LLM)
        elif action_id in [actions.ACTION_GRD_SLM, actions.ACTION_GRD_LLM]:
        # Grade the documents in the active subquery
        # Not checked, may not work
            if state['subqueries']:
                active_sub = state['subqueries'][-1]
                count_rel = 0
                
                # Grade all UNKNOWN docs
                for doc in active_sub['documents']:
                    if doc['relevance'] == "UNKNOWN":
                        use_llm = (action_id == actions.ACTION_GRD_LLM)
                        grade = workers.generate_grade(state, doc['content'], use_llm=use_llm)
                        
                        doc['relevance'] = "RELEVANT" if grade == "Relevant" else "IRRELEVANT"
                        if grade == "Relevant": count_rel += 1

                relevant_indices = []
                for i, doc in enumerate(active_sub['documents']):
                    if doc.get('relevance') == "RELEVANT":
                        relevant_indices.append(f"Doc {i+1} ({doc['title']})")
                
                if relevant_indices:
                    obs = f"Graded docs. {count_rel} relevant: {', '.join(relevant_indices)}"
                else:
                    obs = "Graded docs. None found relevant."

        # [6]: REWRITE (RWT_SLM)
        elif action_id == actions.ACTION_RWT_SLM:
            clean_rw = (argument or "").strip()
            
            # 2. Identify the target (Safe access)
            # Assuming you want the last one, or use a helper like _get_active_subquery(state)
            target_sub = state['subqueries'][-1] if state['subqueries'] else None

            # 3. Execute or Fail
            if len(clean_rw) > 5 and target_sub:
                old_q = target_sub['question']
                target_sub['question'] = clean_rw
                obs = f"Query updated: '{old_q}' -> '{clean_rw}'"
            else:
                obs = "No query rewrite update."

        # [7] or [8]: DECOMPOSITION (DEC_SLM / DEC_LLM)
        elif action_id in [actions.ACTION_DEC_SLM, actions.ACTION_DEC_LLM]:
            plan_text = argument
            if plan_text is None or len(plan_text) < 10:
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
                new_state['subqueries'].extend(new_subs)
                task_preview = "\n".join([f"{i}. {sub['question']}" for i, sub in enumerate(new_subs)])
                obs = f"Decomposed into {len(new_subs)} sub-tasks:\n{task_preview}"
        
            else:
                obs = "No active query to decompose."
        
        # [9]: FAILURE
        elif action_id == actions.ACTION_FAIL:
            obs = "Agent declared failure."
            done = True

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
            argument=final_argument,
            cost=step_cost
        ))

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
       
