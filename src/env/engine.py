from typing import Tuple, List, Optional
import re
from src.env.state import GreenState
from src.agent import actions, workers

# We need a protocol for Retriever or import EphemeralRetriever, but let's assume it's passed in.
# For typing, we can just use Any or specific type if accessible.
from src.env.retriever import EphemeralRetriever

def execute_action(state: GreenState, action_id: int, argument: str, retriever: EphemeralRetriever) -> Tuple[str, bool]:
    """
    Game Engine: Executes the action and transitions the state.
    Returns: (Observation, Done)
    """
    obs = ""
    done = False
    
    # --- 1. ANSWERING (GEN_SLM / GEN_LLM) ---
    if action_id in [actions.ACTION_GEN_SLM, actions.ACTION_GEN_LLM]:
        use_llm = (action_id == actions.ACTION_GEN_LLM)
        # In a real run, workers.generate_answer uses the state. 
        # But wait, workers.generate_answer usually *generates* the text.
        # If the RL agent output the argument "The answer is 42", then that IS the answer.
        # But if the action is "Generate Answer" and argument is "Thinking...", 
        # then we call the worker.
        #
        # DECISION: In PPO, the Policy *IS* the worker. 
        # If Action=Answer, the argument IS the answer.
        # We don't call a separate worker to generate the answer again.
        obs = argument
        done = True 

    # --- 2. RETRIEVAL (RET_KEY / RET_VEC) ---
    elif action_id in [actions.ACTION_RET_KEY, actions.ACTION_RET_VEC]:
        clean_query = argument.strip()
        # Fallback for empty query
        if len(clean_query) < 5 and state['subqueries']:
            clean_query = state['subqueries'][-1]['question']
            
        results = []
        if action_id == actions.ACTION_RET_KEY:
            results = retriever.search_bm25(clean_query, top_k=3)
        else:
            results = retriever.search_dense(clean_query, top_k=3)
            
        obs = f"Found {len(results)} docs."
        
        # Update State
        formatted_docs = []
        for r in results:
            # Retriever returns strings "Title: Content"
            parts = r.split(": ", 1)
            title = parts[0] if len(parts)>1 else "Unknown"
            content = parts[1] if len(parts)>1 else r
            formatted_docs.append({"title": title, "content": content, "relevance": "UNKNOWN"})
            
        if state['subqueries']:
            state['subqueries'][-1]['documents'].extend(formatted_docs)

    # --- 3. GRADING (GRD_SLM) ---
    elif action_id == actions.ACTION_GRD_SLM:
        # Grade the documents in the active subquery
        if state['subqueries']:
            active_sub = state['subqueries'][-1]
            count_rel = 0
            # To be efficient, maybe only grade UNKNOWN ones?
            for doc in active_sub['documents']:
                 if doc['relevance'] == "UNKNOWN":
                     # In PPO, usually we rely on the policy to be the intelligence.
                     # But here the action is "Call Grader". 
                     # So we use the worker (environment oracle).
                     # OR does the policy emit the grade? 
                     # The prompt implies: "Action: 4 (Grade)" -> Environment runs logic.
                     grade = workers.generate_grade(state, doc['content'])
                     doc['relevance'] = "RELEVANT" if grade == "Relevant" else "IRRELEVANT"
                     if grade == "Relevant": count_rel += 1
            obs = f"Graded docs. {count_rel} relevant."
        else:
            obs = "No documents to grade."

    # --- 4. DECOMPOSITION (DEC_SLM / DEC_LLM) ---
    elif action_id in [actions.ACTION_DEC_SLM, actions.ACTION_DEC_LLM]:
         # Logic: Argument is the plan? Or do we call worker?
         # If argument is present and detailed, use it.
         # If argument is just "Decompose", call worker.
         # For RL, let's allow the model to Output the plan in the Argument.
         plan_text = argument
         if len(plan_text) < 10: # Fallback if model was lazy
             use_llm = (action_id == actions.ACTION_DEC_LLM)
             plan_text = workers.generate_plan(state, use_llm=use_llm)
         
         # Execute Plan Parsing
         lines = plan_text.split('\n')
         new_subs = []
         if state['subqueries']:
             parent_id = state['subqueries'][-1]['id']
             for i, line in enumerate(lines):
                clean = line.strip().lstrip('1234567890. ')
                if clean:
                    new_subs.append({
                        "id": f"{parent_id}.{i+1}",
                        "question": clean,
                        "status": "PENDING",
                        "answer": None,
                        "documents": []
                    })
             state['subqueries'].extend(new_subs)
             obs = f"Decomposed into {len(new_subs)} sub-tasks."
         else:
             obs = "No active query to decompose."

    # --- FALLBACK ---
    else:
        obs = "Invalid or No-Op Action."

    return obs, done
