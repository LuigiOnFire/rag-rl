import heapq
import json
import copy
from typing import List, Optional, Any
from src.agent import actions, workers
from src.env.state import GreenState, create_initial_state, SubQuery, Document
from src.env.retriever import EphemeralRetriever
from src.oracle.judge import SoftJudge

# Load Cost Table
try:
    with open("data/meta/cost_table.json", "r") as f:
        COST_TABLE = json.load(f)
except FileNotFoundError:
    print("Warning: cost_table.json not found. Using default costs (1.0).")
    COST_TABLE = {}

def get_cost(action_id):
    return float(COST_TABLE.get(str(action_id), 1.0))

class OracleSearch:
    def __init__(self, retriever: EphemeralRetriever):
        self.retriever = retriever
        self.judge = SoftJudge()

    def get_valid_actions(self, state: GreenState) -> List[int]:
        # Last action string parse "ActionID: ..."
        # For naive loop prevention and basic grammar:
        
        # Always available for now in this search space
        valid = [
            actions.ACTION_RET_KEY, 
            actions.ACTION_RET_VEC, 
            actions.ACTION_DEC_SLM, 
            # actions.ACTION_DEC_LLM, # Optional based on stack
            actions.ACTION_GEN_SLM, 
            actions.ACTION_GEN_LLM,
            actions.ACTION_GRD_SLM
        ]
        
        # Constraints based on state?
        # If no active subquery, maybe force GEN or DEC?
        
        return valid

    def solve(self, start_state_params: Any, max_depth=10) -> Optional[GreenState]:
        # Handle start_state_params which might be the old object or just question string
        # Assuming we pass a GreenState object or similar.
        # But for robustness, let's look at how 02_generate calls it.
        # It calls it with GreenState(question=..., ground_truth=...) (Old class style)
        # We need to adapt.
        
        # Adaptation for 02_generate.py compatibility if it still passes the old class:
        question = ""
        ground_truth = ""
        if hasattr(start_state_params, 'question'):
            question = start_state_params.question
            ground_truth = start_state_params.ground_truth
        else:
            # Fallback if it's already a dict or something else
             question = start_state_params.get('question', '') # If it was a dict
        
        initial_state = create_initial_state(question)
        
        # Priority Queue: (TotalCost, tiebreaker, State)
        # State must be dict, so we can't heapify it directly unless wrapped.
        pq = [(0.0, 0, initial_state)] 
        tiebreaker = 0
        
        best_cost = float('inf')
        solution = None

        while pq:
            cost, _, current_state = heapq.heappop(pq)
            
            if cost >= best_cost:
                continue

            # Check Success
            if current_state['status'] == "SOLVED":
                # We generated a "SOLVED" state.
                # Now we need to verify if the answer is actually correct relative to ground_truth.
                # The "Answer" is in the last subquery answer or implicitly the resolution.
                # In this schema, GEN updates a subquery answer.
                # If the main query is Answered, done.
                
                # Where is the final answer stored? 
                # If main query subquery is answered, that's the result.
                final_sub = current_state['subqueries'][0]
                final_answer = final_sub.get('answer') or ""
                
                if self.judge.judge(final_answer, ground_truth):
                    if cost < best_cost:
                        best_cost = cost
                        solution = current_state
                    continue
                else: 
                    # Wrong answer
                    continue
            
            if len(current_state['recent_history']) >= max_depth:
                continue

            valid_actions = self.get_valid_actions(current_state)
            
            for action_id in valid_actions:
                try:
                    # Deep copy logic for Dict-based state
                    new_state = copy.deepcopy(current_state)
                    
                    # 1. Identify Context (Active Subquery)
                    # Get last ACTIVE or PENDING subquery
                    active_sub = None
                    for sub in reversed(new_state['subqueries']):
                        if sub['status'] in ["ACTIVE", "PENDING"]:
                            active_sub = sub
                            break
                    
                    if not active_sub:
                        # Nothing to work on? Maybe finished?
                        continue
                        
                    active_sub['status'] = "ACTIVE" # Ensure it's active

                    # 2. Generate Argument (Tactics)
                    # Worker needs to see the structured state
                    argument = ""
                    observation = ""
                    
                    if action_id in [actions.ACTION_RET_KEY, actions.ACTION_RET_VEC]:
                        query = workers.generate_search_query(new_state)
                        argument = query
                        
                        docs = []
                        if action_id == actions.ACTION_RET_KEY:
                            docs_text = self.retriever.search_bm25(query)
                        else:
                            docs_text = self.retriever.search_dense(query)
                        
                        # Populate documents
                        new_docs: List[Document] = []
                        for txt in docs_text:
                            # Parse title if possible? Our retrieval returns plain strings "Title: Content"
                            parts = txt.split(": ", 1)
                            title = parts[0] if len(parts) > 1 else "Unknown"
                            content = parts[1] if len(parts) > 1 else txt
                            new_docs.append({"title": title, "content": content, "relevance": "UNKNOWN"})
                        
                        active_sub['documents'].extend(new_docs)
                        observation = f"Found {len(new_docs)} docs."
                        
                    elif action_id in [actions.ACTION_GRD_SLM]:
                         # Grade docs in active_sub
                         count_rel = 0
                         for doc in active_sub['documents']:
                             if doc['relevance'] == "UNKNOWN":
                                 grade = workers.generate_grade(new_state, doc['content'])
                                 doc['relevance'] = "RELEVANT" if grade == "Relevant" else "IRRELEVANT"
                                 if grade == "Relevant": count_rel += 1
                         observation = f"Graded docs. {count_rel} relevant."
                         
                    elif action_id in [actions.ACTION_DEC_SLM]:
                        # Decompose active subquery
                        plan_text = workers.generate_plan(new_state)
                        argument = plan_text
                        
                        # Parse plan (Naive splitting by newline or number)
                        lines = plan_text.split('\n')
                        new_subs = []
                        for i, line in enumerate(lines):
                            clean = line.strip().lstrip('1234567890. ')
                            if clean:
                                new_subs.append({
                                    "id": f"{active_sub['id']}.{i+1}",
                                    "question": clean,
                                    "status": "PENDING",
                                    "answer": None,
                                    "documents": []
                                })
                        
                        # Append new subqueries to the list (after current? or at end?)
                        # Strategy: Add them to end, but they are children of active_sub conceptually.
                        # For flat list processing, appending works.
                        new_state['subqueries'].extend(new_subs)
                        observation = f"Decomposed into {len(new_subs)} sub-tasks."

                    elif action_id in [actions.ACTION_GEN_SLM, actions.ACTION_GEN_LLM]:
                        ans = workers.generate_answer(new_state, use_llm=(action_id==actions.ACTION_GEN_LLM))
                        argument = "Answer Generation"
                        observation = ans
                        
                        # Update Answer
                        active_sub['answer'] = ans
                        active_sub['status'] = "ANSWERED"
                        
                        # If this was the main query (id 1 or first one), mark state as SOLVED
                        if active_sub['id'] == "1" or active_sub == new_state['subqueries'][0]:
                            new_state['status'] = "SOLVED"

                    # Update Metadata
                    step_cost = get_cost(action_id)
                    new_state['total_joules'] += step_cost
                    new_state['recent_history'].append(f"{action_id}: {argument} -> {observation}")
                    
                    tiebreaker += 1
                    heapq.heappush(pq, (cost + step_cost, tiebreaker, new_state))
                    
                except Exception as e:
                    # print(f"Error expanding {action_id}: {e}")
                    continue

        return solution
