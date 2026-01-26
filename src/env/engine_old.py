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
        obs = workers.generate_answer(state, use_llm=use_llm)

        done = True 

    # --- 2. RETRIEVAL (RET_KEY / RET_VEC) ---
    elif action_id in [actions.ACTION_RET_KEY, actions.ACTION_RET_VEC]:
        clean_query = argument.strip()
        # Fallback for empty query
        if len(clean_query) < 5 and state['subqueries']:
            clean_query = state['subqueries'][-1]['question']
            
        results = []
        if action_id == actions.ACTION_RET_KEY:
            results = retriever.search_bm25(clean_query, k=3)
        else:
            results = retriever.search_dense(clean_query, k=3)
            
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

        doc_text_list = []
        for i, doc in enumerate(formatted_docs):
            # Limit length to prevent context overflow (e.g. 100 words per doc)
            preview = doc['content'][:400] + "..." 
            doc_text_list.append(f"[{i+1}] {doc['title']}: {preview}")
        
        # OLD: obs = f"Found {len(results)} docs."
        # NEW:
        obs = f"Found {len(results)} docs:\n" + "\n\n".join(doc_text_list)

    # --- 3. GRADING (GRD_SLM / GRD_LLM) ---
    elif action_id in [actions.ACTION_GRD_SLM, actions.ACTION_GRD_LLM]:
        # Grade the documents in the active subquery
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

    # --- 4. REWRITE (RWT_SLM) ---
    elif action_id == actions.ACTION_RWT_SLM:
        clean_rw = argument.strip()
        if len(clean_rw) > 5 and state['subqueries']:
            old_q = state['subqueries'][-1]['question']
            state['subqueries'][-1]['question'] = clean_rw
            obs = f"Query updated: '{old_q}' -> '{clean_rw}'"
        else:
            obs = "No query rewrite update."

    # --- 5. DECOMPOSITION (DEC_SLM / DEC_LLM) ---
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
             task_preview = "\n".join([f"{i+1}. {sub['question']}" for i, sub in enumerate(new_subs)])
             obs = f"Decomposed into {len(new_subs)} sub-tasks:\n{task_preview}"
     
         else:
             obs = "No active query to decompose."

    # --- 6. FAILURE ---
    elif action_id == actions.ACTION_FAIL:
        obs = "Agent declared failure."
        done = True

    # --- FALLBACK ---
    else:
        obs = "Invalid or No-Op Action."

    return obs, done
