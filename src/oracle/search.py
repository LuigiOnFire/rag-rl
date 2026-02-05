import heapq
import json
import copy
import logging
from typing import List, Optional, Any, Dict, Tuple, TypedDict, NamedTuple
from src.agent import actions, workers
from src.env.state import GreenState, create_initial_state, SubQuery, Document, GreenHistoryItem
from src.env.retriever import EphemeralRetriever
from src.oracle.judge import SoftJudge
from src.env.engine import GreenEngine

logging.getLogger().setLevel(logging.DEBUG)

trace_logger = logging.getLogger("LLM_TRACE")
trace_logger.addHandler(logging.NullHandler())

# Load Cost Table
try:
    with open("data/meta/cost_table.json", "r") as f:
        COST_TABLE = json.load(f)
except FileNotFoundError:
    logging.warning("cost_table.json not found. Using default costs (1.0).")
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
        self.engine = GreenEngine(retriever=retriever)
        self.retriever = retriever
        self.judge = SoftJudge()
        # If set, the first action will always be DEC_LLM
        self.force_decompose = False

    def get_valid_actions(self, state: GreenState) -> List[int]:
        # 1. Parse the last action from history to check context
        last_action_id = None
        if state['history']:
            # New dict-based GreenStep
            last_action_id = state['history'][-1]['action_id']

        # 2. START OF TURN (Don't start with grade or rewrite)
        if not state['history']:
            if self.force_decompose:
                return [
                    actions.ACTION_DEC_LLM,
                ]
            else:
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

    def solve(self, start_state_params: Any, ground_truth: str, max_depth=10) -> Tuple[Optional[GreenState], Dict[str, Any]]:
        # Adaptation for 02_generate.py compatibility
        logging.debug("Starting OracleSearch.solve()")
        initial_state: GreenState
        
        if isinstance(start_state_params, dict) and "question" in start_state_params:
             # It is already a GreenState dict
             # type ignore used because mypy/pylance struggles with deepcopy of TypedDict vs dict
             initial_state = copy.deepcopy(start_state_params) # type: ignore
        else:
            raise ValueError("start_state_params must be a GreenState dict.")
    
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
            logging.debug("RESTARTING LOOP: WE SHOULD SEE MANY NODES EXPANDED")
            logging.debug(f"Priority Queue Size: {len(pq)}")
            cost, _, current_node = heapq.heappop(pq)
            # MUCH OF THIS FUNCTIONALITY WILL BE MOVED TO engine.py
            current_state = current_node.state
            nodes_expanded += 1
            
           # Log this step (snapshot summary)
            logging.debug(f"Step {nodes_expanded}: cost={cost}, history_len={len(current_state['history'])}, last_action={current_state['history'][-1]['action_id'] if current_state['history'] else 'START'}, status={current_state.get('status', 'SOLVING')}")
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
            logging.debug(f"Current state status: {current_state['status']}")
            if current_state['status'] == "SOLVED":
                # We generated a "SOLVED" state.
                logging.debug("Entering SOLVED state evaluation.")
                final_answer = current_state.get('answer')
                question  = current_state.get('question')

                if final_answer is None:
                    logging.debug("No final answer found in SOLVED state; continuing search.")
                    continue

                logging.debug(f"Final answer to judge: {final_answer}")
                logging.debug(f"Solved with answer: {final_answer}")                
                is_correct, reason = self.judge.judge(final_answer, ground_truth, question)

                trace_logger.debug(f"JUDGE LOG -- Q: {question} | A: {final_answer} | GT: {ground_truth} | Correct: {is_correct} | Reason: {reason}")

                
                if is_correct:
                    # Old Logic would keep looking for cheaper solutions
                    # if cost < best_cost:
                    #     best_cost = cost
                    #     solution_node = current_node

                    # New logic, do not continue searching
                    # Just take this one and move on 
                    solution_node = current_node
                    break
                else: 
                    # Wrong answer
                    logging.debug(f"Wrong answer: {final_answer}, Reason: {reason}")
                    continue
            
            if len(current_state['history']) >= max_depth:
                step_info["drop_reason"] = "max_depth"
                continue

            valid_actions = self.get_valid_actions(current_state)

            # We will tell the engine to try each action
            for action_id in valid_actions:
                try:
                    # Deep copy logic for Dict-based state
                    # new_state = copy.deepcopy(current_state)
                    
                    # 1. Identify Context (Active Subquery)
                    # Get last ACTIVE or PENDING subquery                    

                    # 2. Generate Argument (Tactics)
                    # Worker needs to see the structured state
                    argument = ""
                    observation = ""

                    # Get the next step from the engine
                    logging.debug(f"Expanding action {action_id} from current state.")
                    new_state = self.engine.step(current_state, action_id, argument=None)

                    last_step = new_state['history'][-1]
                    step_cost = last_step['cost']        

                    tiebreaker += 1           

                    # Create our search node
                    new_node = SearchNode(
                        state=new_state,
                        parent=current_node,
                        action_item=last_step
                    )

                    # Push this node to our heap
                    heapq.heappush(pq, (cost + step_cost, tiebreaker, new_node))
                
                except Exception as e:
                    logging.error(f"Error expanding {action_id}: {e}")
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
            logging.debug(f"Reconstructing SFT trajectory with {len(path_nodes)-1} steps.")
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
        logging.debug(f"Debug Info: {debug_info}")
        return final_state, debug_info

