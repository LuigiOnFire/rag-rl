import os
import logging

import uuid
from typing import Optional, Tuple, Dict, Any
from codecarbon import EmissionsTracker
from contextlib import contextmanager
import time


from src.env import state
import copy
import json
import hashlib
from src.env.state import GreenState, GreenHistoryItem, get_active_subquery
from src.agent import actions, workers
from src.env.retriever import EphemeralRetriever, GlobalRetriever
from typing import Union  # <--- Add this line!

# Get the cost table
try:
    with open("data/meta/cost_table.json", "r") as f:
        COST_TABLE = json.load(f)
except FileNotFoundError:
    print("Warning: cost_table.json not found. Using default costs (1.0).")
    COST_TABLE = {}

trace_logger = logging.getLogger("LLM_TRACE")
trace_logger.addHandler(logging.NullHandler())

def _hash_doc(doc: dict) -> str:
    """Creates a deterministic hash using both title and text to avoid the chunking trap."""
    title = doc.get("title", "")
    text = doc.get("content", "")
    
    # Strip whitespace to prevent formatting artifacts from breaking the hash
    clean_string = " ".join(f"{title} {text}".split())
    return hashlib.md5(clean_string.encode('utf-8')).hexdigest()

# A class to encapsulate the engine logic and manage the state
class GreenEngine:
    def __init__(self, retriever: Union[EphemeralRetriever, GlobalRetriever]):
        self.retriever = retriever

        hidden_slurm_vars = {}
        for key in list(os.environ.keys()):
            if key.startswith("SLURM"):
                hidden_slurm_vars[key] = os.environ.pop(key)

        self.tracker = EmissionsTracker(
            measure_power_secs=0.5,  # CRITICAL: Forces sampling every 0.5s for fast micro-actions
            save_to_file=False,      # Prevents CodeCarbon from spamming your disk with CSVs
            log_level="error"        # Suppresses background print statements
        )

        self.tracker.start()
    
    def get_cost(self, action_id: int) -> float:
        return COST_TABLE.get(str(action_id), 1.0)

    @contextmanager
    def track_energy(self, action_name: str):
        """Context manager to securely wrap tracking around an execution block."""
        try:
            task_name = f"{action_name}_{uuid.uuid4().hex[:6]}"
            self.tracker.start_task(task_name)

            # Yield a dictionary so we can pass the cost back out to the main function
            metrics = {}
            yield metrics
        finally:
            task = self.tracker.stop_task(task_name)
            # Robust extraction matching
            if task and hasattr(task, 'emissions_data'):
                energy_kwh = task.emissions_data.energy_consumed
            elif task and hasattr(task, 'energy_consumed'):
                energy_kwh = task.energy_consumed
            else:
                energy_kwh = 0.0
                
            metrics['cost'] = energy_kwh * 3_600_000
    
    
    def step(self, state: GreenState, action_id: int, argument: Optional[str] = None, task_type: str = "qa") -> GreenState:
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

        # Track the active subquery index so we can update the state directly
        active_idx = -1
        active_subquery = None

        # Iterate in natural order so execution matches the plan order
        subqueries = new_state.get('subqueries', [])
        for i, sub in enumerate(subqueries):
            if sub['status'] in ["ACTIVE", "PENDING"]:
                active_idx = i
                active_subquery = sub
                # Side effect: Mark as ACTIVE immediately in the state
                new_state['subqueries'][i]['status'] = "ACTIVE"
                break

        # Execute the action using the engine function
        # We don't use the obs or argument for generation
        obs = ""

        logging.debug(f"Attempting Action ID {action_id} on State with status: {new_state['status']}")

        action_name_str = actions.get_action_name(action_id)


        start_time = time.perf_counter()


        with self.track_energy(action_name_str) as metrics:
            # --- [0] or [1]: ANSWERING (GEN_SLM / GEN_LLM) ---
            if action_id in [actions.ACTION_GEN_SLM, actions.ACTION_GEN_LLM]:
                use_llm = (action_id == actions.ACTION_GEN_LLM)

                # Hold on, doesn't this depend on whether or not we have a subquery too? We need to know what to pass here.
                # This function will figure out what the active query is on its own
                answer, size_metrics = workers.generate_answer(new_state, use_llm=use_llm, task_type=task_type)

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
                    argument, size_metrics = workers.generate_query_for_keyword_search(new_state, use_llm=False)
                    raw_docs = self.retriever.search_bm25(argument)
                else:
                    argument, size_metrics = workers.generate_query_for_vector_search(new_state, use_llm=True)
                    raw_docs = self.retriever.search_dense(argument)

                # Update prev_searches in new_state
                new_state['prev_searches'].append(argument)
                
                # Format & Update State
                formatted_docs = self._format_docs(raw_docs)

                # 1. Determine which document list we are currently building
                target_doc_list = new_state['documents']
                
                # 2. Build a set of hashes we already have in that list
                existing_hashes = {_hash_doc(doc) for doc in target_doc_list}
                
                # 3. Filter for unique documents
                unique_docs = []
                for doc in formatted_docs:
                    doc_hash = _hash_doc(doc)
                    if doc_hash not in existing_hashes:
                        unique_docs.append(doc)
                        existing_hashes.add(doc_hash)

                # 4. Update State with ONLY the unique docs
                target_doc_list.extend(unique_docs)
                
                search_type = "Keyword Search" if action_id == actions.ACTION_RET_KEY else "Vector Search"
                obs = f"[{search_type} executed for: '{argument}'] Found {len(formatted_docs)} docs. Added {len(unique_docs)} unique docs to context."

            # --- [4] or [5]: REASON (RSN_SLM / RSN_LLM)
            elif action_id in [actions.ACTION_RSN_SLM, actions.ACTION_RSN_LLM]:
                # Generate either a short-term plan (SLM) or a long-term strategy (LLM)
                use_llm = (action_id == actions.ACTION_RSN_LLM)
                # Pass the updated state (with any active-subquery set) to the reasoning worker
                plan_text, size_metrics = workers.generate_reasoning(new_state, use_llm)

                # Overwrite strategy if we used LLM, otherwise set the short-term plan
                if use_llm:
                    new_state['strategy'] = plan_text
                    obs = f"Generated strategy: {plan_text}"
                else:
                    new_state['plan'] = plan_text
                    obs = f"Generated plan: {plan_text}"


            # [6] or [7]: DECOMPOSITION (DEC_LLM / DEC_RSN)
            elif action_id in [actions.ACTION_DEC_LLM, actions.ACTION_DEC_RSN]:
                reasoning_mode = (action_id == actions.ACTION_DEC_RSN)
                plan_text, size_metrics = workers.generate_plan(state, reasoning_mode)
                
                # Format the plan into subqueries
                # Not sure how well this is going to work but we can iterate

                # Previously this would recursively decompose the active subquery
                # I'm going to change the logic here
                # Now we always decompose the MAIN query, overwriting any existing subqueries
                if "---SUBQUERIES---" in plan_text:
                    reasoning_text, subquery_text = plan_text.split("---SUBQUERIES---", 1)
                else:
                    subquery_text = plan_text

                lines = subquery_text.strip().split('\n')
                new_subs = []

                for i, line in enumerate(lines):
                    clean = line.strip().lstrip('1234567890.-* ')
                    if clean:
                        new_subs.append({
                            "id": f"{i}",
                            "question": clean,
                            "status": "PENDING",
                            "answer": None
                        })

                new_state['subqueries'] = new_subs
                task_preview = "\n".join([f"{i}. {sub['question']}" for i, sub in enumerate(new_subs)])
                obs = f"Decomposed into {len(new_subs)} sub-tasks:\n{task_preview}"
                    
            # [8]: FAILURE
            elif action_id == actions.ACTION_FAIL:
                obs = "Agent declared failure."
                new_state['status'] = "FAILED" # <--- Persist the failure

            # --- FALLBACK ---
            else:
                obs = "Invalid or No-Op Action."

        end_time = time.perf_counter()
        action_duration = end_time - start_time

        actual_input_size = size_metrics.get("input_size", 0) if 'size_metrics' in locals() else 0
        actual_output_size = size_metrics.get("output_size", 0) if 'size_metrics' in locals() else 0

        # History and Cost Update
        step_cost = metrics['cost']
        new_state['total_joules'] += step_cost

        new_state['history'].append(GreenHistoryItem(
            action_id=action_id,
            action_name=actions.get_action_name(action_id),
            observation=obs,
            argument=argument if argument is not None else "",
            cost=step_cost,
            input_state_size=actual_input_size, 
            output_state_size=actual_output_size,
            duration_seconds=action_duration   
        ))

        logging.debug(f"Before Step End: New State status: {new_state['status']}, action: {action_id}, observation: {obs}")
        return new_state

    #     DEPRECATED

    #     # [4] or [5]: GRADING (GRD_SLM / GRD_LLM)
    #     elif action_id in [actions.ACTION_GRD_SLM, actions.ACTION_GRD_LLM]:
    #     # Grade the documents in the active subquery
    #     # Not checked, may not work
    #         count_rel = 0
    #         target_docs = active_subquery['documents'] if active_subquery is not None else new_state.get('documents', [])

    #         if not target_docs:
    #             obs = "No documents to grade."
    #         else:
    #             logging.debug(f"Grading {len(target_docs)} documents for relevance.")
    #             use_llm = (action_id == actions.ACTION_GRD_LLM)
                
    #             for doc in target_docs:
    #                 grade = workers.generate_grade(new_state, doc["content"], use_llm=use_llm)
    #                 doc['relevance'] = "RELEVANT" if grade == "Relevant" else "IRRELEVANT"
    #                 if grade == "Relevant": count_rel += 1

    #             relevant_indices = []
    #             for i, doc in enumerate(target_docs):
    #                 if doc.get('relevance') == "RELEVANT":
    #                     relevant_indices.append(f"Doc {i+1} ({doc['title']})")
                        
    #             if relevant_indices:
    #                 obs = f"Graded docs. {count_rel} relevant: {', '.join(relevant_indices)}"
    #             else:
    #                 obs = "Graded docs. None found relevant."

                    

    #    # [6]: REWRITE (RWT_SLM)
    #     elif action_id == actions.ACTION_RWT_SLM:           
    #         if active_subquery is not None:
    #             old_query = active_subquery.get("question", "")
                
    #             # Generate the new, specific question using past answers
    #             new_query = workers.generate_rewrite(new_state)
                
    #             if new_query and new_query.lower() != old_query.lower():
    #                 active_subquery["question"] = new_query
    #                 obs = f"Rewrote pending sub-query to: '{new_query}'"
    #             else:
    #                 obs = "Rewrite deemed unnecessary or failed. Kept original query."
    #         else:
    #             obs = "Action failed. No active sub-query to rewrite."

    
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
       
