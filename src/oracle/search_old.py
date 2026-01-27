import heapq
import json
import copy
from typing import List, Optional, Any, Dict, Tuple, TypedDict, NamedTuple
from src.agent import actions, workers
from src.env.state import GreenState, create_initial_state, SubQuery, Document, GreenHistoryItem
from src.env.retriever import EphemeralRetriever
from src.oracle.judge import SoftJudge
from src.env.engine_old import GreenEngine

# Load Cost Table
try:
    with open("data/meta/cost_table.json", "r") as f:
        COST_TABLE = json.load(f)
except FileNotFoundError:
    print("Warning: cost_table.json not found. Using default costs (1.0).")
    COST_TABLE = {}

def get_cost(action_id):
    return float(COST_TABLE.get(str(action_id), 1.0))

class SearchNode(NamedTuple):
    state: GreenState
    parent: Optional['SearchNode']
    action_item: Optional[GreenHistoryItem]
    # We store the parent link to reconstruct the full pre_state trajectory

class OracleSearch:
    def __init__(self, retriever: EphemeralRetriever):
        self.retriever = retriever
        self.judge = SoftJudge()

    def get_valid_actions(self, state: GreenState) -> List[int]:
        # 1. Parse the last action from history to check context
        last_action_id = None
        if state['history']:
            # New dict-based GreenStep
            last_action_id = state['history'][-1]['action_id']

        # 2. START OF TURN (Don't start with grade or rewrite)
        if not state['history']:
            return [
                actions.ACTION_RET_KEY, 
                actions.ACTION_RET_VEC, 
                actions.ACTION_DEC_SLM,
                actions.ACTION_DEC_LLM,
                actions.ACTION_GEN_SLM, 
                actions.ACTION_GEN_LLM
            ]

        # 3. MIDDLE OF TURN
        else:
            # Base actions always available
            valid = [
                actions.ACTION_RET_KEY, 
                actions.ACTION_RET_VEC, 
                actions.ACTION_DEC_SLM,
                actions.ACTION_DEC_LLM, 
                actions.ACTION_GEN_SLM, 
                actions.ACTION_GEN_LLM
            ]
            
            # LOGIC FIX: Only allow GRADE if we just RETRIEVED
            # otherwise, what are we grading?
            retrieval_ids = [actions.ACTION_RET_KEY, actions.ACTION_RET_VEC]
            
            if last_action_id in retrieval_ids:
                valid.append(actions.ACTION_GRD_SLM)
                
            # LOGIC FIX: Only allow REWRITE if we just GRADED
            # (Optional, but saves energy)
            if last_action_id == actions.ACTION_GRD_SLM:
                valid.append(actions.ACTION_RWT_SLM)

            return valid

    def solve(self, start_state_params: Any, max_depth=10) -> Tuple[Optional[GreenState], Dict[str, Any]]:
        # Adaptation for 02_generate.py compatibility
        initial_state: GreenState
        
        if isinstance(start_state_params, dict) and "question" in start_state_params:
             # It is already a GreenState dict
             # type ignore used because mypy/pylance struggles with deepcopy of TypedDict vs dict
             initial_state = copy.deepcopy(start_state_params) # type: ignore
        else:
             # Legacy or plain object support
             question = ""
             ground_truth = ""             
             if hasattr(start_state_params, 'question'):
                 question = getattr(start_state_params, 'question')
                 ground_truth = getattr(start_state_params, 'ground_truth', "")
             elif isinstance(start_state_params, dict):
                 question = start_state_params.get('question', '')
            
             initial_state = create_initial_state(question, ground_truth)
        
        # Ensure ground_truth variable is bound for the loop
        ground_truth = initial_state['ground_truth']

        # Priority Queue: (TotalCost, tiebreaker, SearchNode)
        # We wrap the state in a SearchNode to track the path (parent)
        root_node = SearchNode(state=initial_state, parent=None, action_item=None)

        # This is our heap/priority queue
        # [cost, tiebreaker, node]
        pq = [(0.0, 0, root_node)] 
        tiebreaker = 0
        
        best_cost = float('inf')
        solution_node: Optional[SearchNode] = None
        
        # Debug / Search Trace
        search_trace = []
        nodes_expanded = 0

        while pq:
            cost, _, current_node = heapq.heappop(pq)
            current_state = current_node.state
            nodes_expanded += 1
            
            # Log this step (snapshot summary)
            step_info = {
                "step_idx": nodes_expanded,
                "cost": cost,
                "history_len": len(current_state['history']),
                "last_action": current_state['history'][-1]['action_id'] if current_state['history'] else "START",
                "status": current_state.get('status', 'SOLVING')
            }
            search_trace.append(step_info)
            
            if cost >= best_cost:
                continue

            # Check Success
            if current_state['status'] == "SOLVED":
                # We generated a "SOLVED" state.
                final_sub = current_state['subqueries'][0]
                final_answer = final_sub.get('answer') or ""
                
                is_correct, reason = self.judge.judge(final_answer, ground_truth)
                step_info["judge_verdict"] = is_correct
                step_info["judge_reason"] = reason
                
                if is_correct:
                    if cost < best_cost:
                        best_cost = cost
                        solution_node = current_node
                        # Update the state in the node with judge log
                        solution_node.state['judge_log'] = reason
                    continue
                else: 
                    # Wrong answer
                    continue
            
            if len(current_state['history']) >= max_depth:
                step_info["drop_reason"] = "max_depth"
                continue

            valid_actions = self.get_valid_actions(current_state)
            
            for action_id in valid_actions:
                try:
                    # Deep copy logic for Dict-based state
                    new_state = copy.deepcopy(current_state)
                    
                    # 1. Identify Context (Active Subquery)
                    # Get last ACTIVE or PENDING subquery
                    active_subquery = None
                    for sub in reversed(new_state['subqueries']):
                        if sub['status'] in ["ACTIVE", "PENDING"]:
                            active_subquery = sub
                            break
                    
                    if not active_subquery:
                        # Nothing to work on? Maybe finished?
                        continue
                        
                    active_subquery['status'] = "ACTIVE" # Ensure it's active

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
                            assert action_id == actions.ACTION_RET_VEC # should be guranteed by if condition
                            docs_text = self.retriever.search_dense(query)
                        
                        # Populate documents
                        new_docs: List[Document] = []
                        for txt in docs_text:
                            # Parse title if possible? Our retrieval returns plain strings "Title: Content"
                            parts = txt.split(": ", 1)
                            title = parts[0] if len(parts) > 1 else "Unknown"
                            content = parts[1] if len(parts) > 1 else txt
                            new_docs.append({"title": title, "content": content, "relevance": "UNKNOWN"})
                        
                        active_subquery['documents'].extend(new_docs)
                        observation = f"Found {len(new_docs)} docs."
                        
                    elif action_id in [actions.ACTION_GRD_SLM]:
                         # Grade docs in active_sub
                         count_rel = 0
                         for doc in active_subquery['documents']:
                             if doc['relevance'] == "UNKNOWN":
                                 grade = workers.generate_grade(new_state, doc['content'])
                                 doc['relevance'] = "RELEVANT" if grade == "Relevant" else "IRRELEVANT"
                                 if grade == "Relevant": count_rel += 1
                         observation = f"Graded docs. {count_rel} relevant."
                         
                    elif action_id in [actions.ACTION_DEC_SLM, actions.ACTION_DEC_LLM]:
                        # Decompose active subquery
                        plan_text = workers.generate_plan(new_state, use_llm=(action_id==actions.ACTION_DEC_LLM))
                        argument = plan_text
                        
                        # Parse plan (Naive splitting by newline or number)
                        lines = plan_text.split('\n')
                        new_subs = []
                        for i, line in enumerate(lines):
                            clean = line.strip().lstrip('1234567890. ')
                            if clean:
                                new_subs.append({
                                    "id": f"{active_subquery['id']}.{i+1}",
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
                        argument = ""
                        observation = ans
                        
                        # Update Answer
                        active_subquery['answer'] = ans
                        active_subquery['status'] = "ANSWERED"
                        
                        # If this was the main query (id 1 or first one), mark state as SOLVED
                        if active_subquery['id'] == "1" or active_subquery == new_state['subqueries'][0]:
                            new_state['status'] = "SOLVED"

                    # Update Metadata
                    step_cost = get_cost(action_id)
                    new_state['total_joules'] += step_cost
                    
                    # Create Lightweight History Item
                    history_item: GreenHistoryItem = {
                        "action_id": action_id,
                        "action_name": actions.get_action_name(action_id),
                        "argument": argument,
                        "observation": observation,
                        "cost": step_cost
                    }
                    new_state['history'].append(history_item)
                    
                    # Create New Node
                    new_node = SearchNode(state=new_state, parent=current_node, action_item=history_item)
                    
                    tiebreaker += 1
                    heapq.heappush(pq, (cost + step_cost, tiebreaker, new_node))
                    
                except Exception as e:
                    # print(f"Error expanding {action_id}: {e}")
                    continue

        # Reconstruct Trajectory from Solution Node
        sft_trajectory = []
        final_state = None
        
        if solution_node:
            final_state = solution_node.state
            
            # Backtrack
            curr = solution_node
            path_nodes = []
            while curr:
                path_nodes.append(curr)
                curr = curr.parent
            
            # Reverse to get chronological order (Start -> End)
            path_nodes.reverse()
            
            # Now build SFT steps: (Pre-State from parent) -> Action
            # path_nodes[0] is Root (Start State, no action leading to it)
            # path_nodes[1] is State 1 (created by Action 1 from State 0)
            
            for i in range(1, len(path_nodes)):
                node = path_nodes[i]
                parent = path_nodes[i-1]
                
                # The action that took us from Parent -> Node
                act = node.action_item 
                # The state input was the Parent's state
                # Create a copy and remove ground_truth to prevent data leakage in training
                pre_state = copy.copy(parent.state)
                if 'ground_truth' in pre_state:
                    del pre_state['ground_truth']
                
                if act:
                    sft_trajectory.append({
                        "step_id": i-1,
                        "pre_state": pre_state, # Saved snapshot
                        "action_id": act['action_id'],
                        "argument": act['argument'],
                        "observation": act['observation']
                    })

        debug_info = {
            "nodes_expanded": nodes_expanded,
            "best_cost": best_cost if best_cost != float('inf') else None,
            "solved": solution_node is not None,
            "trace": search_trace,
            "sft_trajectory": sft_trajectory
        }
        return final_state, debug_info

