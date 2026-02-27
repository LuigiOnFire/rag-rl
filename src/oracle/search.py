import heapq
import json
import copy
import logging
from typing import List, Optional, Any, Dict, Tuple, TypedDict, NamedTuple
from src.agent import actions, workers
from src.env.state import GreenState, GreenHistoryItem
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

def get_cost(action_id: int) -> float:
    return float(COST_TABLE.get(str(action_id), 1.0))

# Average number of subqueries assumed for decompose strategies when ranking cost.
# Tune this to match your typical multi-hop question depth.
AVG_DECOMPOSE_SUBQUERIES = 3

def strategy_cost(strategy: list) -> float:
    """
    Estimate the total expected Joule cost for a WaterfallOracle strategy.
    
    - int entries: a single action at its calibrated cost.
    - tuple entries: a repeat block executed once per subquery, so the cost
      of the tuple body is multiplied by AVG_DECOMPOSE_SUBQUERIES.
    """
    total = 0.0
    for entry in strategy:
        if isinstance(entry, tuple):
            total += AVG_DECOMPOSE_SUBQUERIES * sum(get_cost(a) for a in entry)
        else:
            total += get_cost(entry)
    return total

class OracleInterface:
    """Common interface to ensure both solvers behave the same way."""
    def solve(self, state: GreenState, ground_truth: str) -> Tuple[Optional[GreenState], Dict]:
        raise NotImplementedError

class SearchNode(NamedTuple):
    state: GreenState
    parent: Optional['SearchNode']
    action_item: Optional[GreenHistoryItem]
    # We store the parent link to reconstruct the full pre_state trajectory

class OracleSearch(OracleInterface):
    def __init__(self, retriever: EphemeralRetriever):
        self.engine = GreenEngine(retriever=retriever)
        self.retriever = retriever
        self.judge = SoftJudge()
        # If set, the first action will always be DEC_LLM
        self.force_decompose = False

        # Maybe a temporary flag, if we have already decompose, never do it again in the same episode
        self.decompose_once = True

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
            # FOR DEBUGGING
            # Base actions always available
            valid = [
                actions.ACTION_RET_KEY, 
                actions.ACTION_RET_VEC, 
                actions.ACTION_GEN_SLM, 
                actions.ACTION_GEN_LLM
            ]

            # LOGIC FIX: Only allow GRADE if we just RETRIEVED
            # otherwise, what are we grading?
            retrieval_ids = [actions.ACTION_RET_KEY, actions.ACTION_RET_VEC]
            
            # if last_action_id in retrieval_ids:
            #     valid.append(actions.ACTION_GRD_SLM)
                
            # LOGIC FIX: Only allow REWRITE if we just GRADED
            # (Optional, but saves energy)
            if last_action_id == actions.ACTION_GRD_SLM:
                valid.append(actions.ACTION_RWT_SLM)
            
            # If we have decompose once active, don't allow further decomposition
            if not self.decompose_once:
                valid.append(actions.ACTION_DEC_SLM)
                valid.append(actions.ACTION_DEC_LLM)

            return valid

    def solve(self, start_state_params: GreenState, ground_truth: str, max_depth=10) -> Tuple[Optional[GreenState], Dict[str, Any]]:
        # Adaptation for 02_generate.py compatibility
        logging.debug("Starting OracleSearch.solve()")
        initial_state: GreenState
        
        if isinstance(start_state_params, dict) and "question" in start_state_params:
             # It is already a GreenState dict
             # type ignore used because mypy/pylance struggles with deepcopy of TypedDict vs dict
             initial_state = copy.deepcopy(start_state_params)
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
            cost, _, current_node = heapq.heappop(pq)
            # MUCH OF THIS FUNCTIONALITY WILL BE MOVED TO engine.py
            current_state = current_node.state
            nodes_expanded += 1
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
                # We generated a "SOLVED" state - this is terminal, don't expand further
                logging.debug("Entering SOLVED state evaluation.")
                final_answer = current_state.get('answer')
                question  = current_state.get('question')

                if final_answer is None:
                    logging.debug("No final answer found in SOLVED state; continuing search.")
                    continue

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
                    # Wrong answer - this is a dead end, don't expand
                    logging.debug(f"Wrong answer: {final_answer}, Reason: {reason}")
                    continue
            
            if len(current_state['history']) >= max_depth:
                step_info["drop_reason"] = "max_depth"
                continue

            valid_actions = self.get_valid_actions(current_state)

            # We will tell the engine to try each action
            for action_id in valid_actions:
                try:
                    trace_logger.debug(f"The search is taking action {action_id} from state with history length {len(current_state['history'])} and status {current_state.get('status', 'UNKNOWN')}")

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

class WaterfallOracle(OracleInterface):
    """
    The previous implementation would attempt to do a shortest path search of the tree
    to find the most efficient solution was simply too expensive to run, especially in 
    the case of decompose actions which can generate a large number of new states. 
    This new implementation will simply follow a fixed policy of actions, without any 
    backtracking or search. This is a "waterfall" approach, where we go 
    step by step through a predefined sequence of actions until we reach a solution 
    or exhaust our options.
    """
    def __init__(self, retriever: EphemeralRetriever):
        self.engine = GreenEngine(retriever=retriever)
        self.judge = SoftJudge()

    def solve(self, start_state_params: GreenState, ground_truth: str, max_depth=10) -> Tuple[Optional[GreenState], Dict[str, Any]]:
        """
        Tried a handful of strategies with increasing cost.
        These strategies are handwritten solutions.
        I will monitor the performance of these strategies and add more as needed.
        """
        if isinstance(start_state_params, dict) and "question" in start_state_params:
            template_state = copy.deepcopy(start_state_params)
        else:
            raise ValueError("start_state_params must be a GreenState dict.")
        
        # Then, run through each strategy
        # The strategies are defined as class variables
        for strategy_idx, strategy in enumerate(self.STRATEGIES):
            logging.debug(f"Waterfall: Attempting Strategy {strategy_idx} - {strategy}")
            
            # First, copy the state
            current_state = copy.deepcopy(template_state)

            pre_states= []

            for action_id in strategy:
                # ── Repeat block: tuple means "loop until no active subquery" ──
                # e.g. (ACTION_RET_KEY, ACTION_GEN_SLM) will keep cycling
                # through those actions until get_active_subquery() returns None.
                if isinstance(action_id, tuple):
                    from src.env.state import get_active_subquery
                    repeat_actions = action_id
                    while get_active_subquery(current_state) is not None:
                        for sub_action in repeat_actions:
                            pre_states.append(copy.deepcopy(current_state))
                            current_state = self.engine.step(current_state, sub_action, argument=None)
                            if current_state['status'] in ("SOLVED", "FAILED"):
                                break
                        if current_state['status'] in ("SOLVED", "FAILED"):
                            break
                    # After the loop, fall through to the SOLVED check below
                    # by treating it as a no-op single step (skip the normal step)
                else:
                    # Normal single action
                    pre_states.append(copy.deepcopy(current_state))
                    current_state = self.engine.step(current_state, action_id, argument=None)

                # Check if solved
                if current_state['status'] == "SOLVED":
                    final_answer = current_state.get('answer')
                    question  = current_state.get('question')

                    if final_answer is None:
                        continue

                    # After running each streategy, check if the problem was solved
                    is_correct, reason = self.judge.judge(final_answer, ground_truth, question)
                    
                    # If it was solved, return the solution and the trajectory
                    if is_correct:
                        # build the SFT trajectory for this successful strategy
                        sft_trajectory = self._build_trajectory(pre_states, current_state['history'])

                        return current_state, {"solved": True, 
                                               "strategy": strategy,
                                               "sft_trajectory": sft_trajectory,
                                               }
                    else:
                        # If it was not solved, move on to the next strategy
                        break # Move on to the next strategy
        
    
        return None, {"solved": False, "sft_trajectory": []}
    
    def _build_trajectory(self, pre_states: List[GreenState], history: List[GreenHistoryItem]) -> List[Dict]:
        """
        Zips the pre_states with the resulting history items to create training examples.
        """
        sft_trajectory = []
        
        # Ensure they align (they should, unless the engine failed to append to history)
        limit = min(len(pre_states), len(history))
        
        for i in range(limit):
            pre_state = pre_states[i]
            act = history[i]
            
            # Remove ground truth to prevent data leakage in training, just like OracleSearch did
            if 'ground_truth' in pre_state:
                del pre_state['ground_truth']
                
            sft_trajectory.append({
                "step_id": i,
                "pre_state": pre_state, 
                "action_id": act['action_id'],
                "argument": act['argument'],
                "observation": act['observation']
            })
            
        return sft_trajectory
    
    STRATEGIES = [
        # Strategy 1.1: Try to solve directly with SLM generation
        [actions.ACTION_GEN_SLM],

        # Strategy 1.2: Same but with LLM generation
        [actions.ACTION_GEN_LLM],

        # Strategy 2.1: Key search, then generate with SLM
        [
            actions.ACTION_RET_KEY, 
            actions.ACTION_GEN_SLM
        ],

        # Strategy 2.2: Key search, then generate with LLM
        [
            actions.ACTION_RET_KEY, 
            actions.ACTION_GEN_LLM
        ],

        # Strategy 2.3 Vector search, then generate with SLM
        [
            actions.ACTION_RET_VEC, 
            actions.ACTION_GEN_SLM
        ],

        # Strategy 2.4 Vector search, then generate with LLM
        [
            actions.ACTION_RET_VEC, 
            actions.ACTION_GEN_LLM
        ],

        # Strategy 3.1: Decompose with SLM, then generate with SLM
        [
            actions.ACTION_DEC_SLM,
            (actions.ACTION_GEN_SLM,),  # repeat until no active subquery
            actions.ACTION_GEN_SLM,     # final synthesis answer
        ],

        # Strategy 3.2: Decompose with SLM, then generate with LLM
        [
            actions.ACTION_DEC_SLM,
            (actions.ACTION_GEN_LLM,),  # repeat until no active subquery
            actions.ACTION_GEN_LLM,     # final synthesis answer
        ],

        # Strategy 4.1: Decompose with SLM, then retrieve with keyword and generate with SLM for each subproblem
        [
            actions.ACTION_DEC_SLM,
            (actions.ACTION_RET_KEY, actions.ACTION_GEN_SLM),  # repeat until no active subquery
            actions.ACTION_GEN_SLM,     # final synthesis answer
        ],

        # Strategy 4.2: Decompose with SLM, then retrieve with keyword and generate with LLM for each subproblem
        [
            actions.ACTION_DEC_SLM,
            (actions.ACTION_RET_KEY, actions.ACTION_GEN_LLM),  # repeat until no active subquery
            actions.ACTION_GEN_LLM,     # final synthesis answer
        ],

        # Strategy 4.3: Decompose with SLM, then retrieve with vector and generate with SLM for each subproblem
        [
            actions.ACTION_DEC_SLM,
            (actions.ACTION_RET_VEC, actions.ACTION_GEN_SLM),  # repeat until no active subquery
            actions.ACTION_GEN_SLM,     # final synthesis answer
        ],

        # Strategy 4.4: Decompose with SLM, then retrieve with vector and generate with LLM for each subproblem
        [
            actions.ACTION_DEC_SLM,
            (actions.ACTION_RET_VEC, actions.ACTION_GEN_LLM),  # repeat until no active subquery
            actions.ACTION_GEN_LLM,     # final synthesis answer
        ],
        # Strategy 5.1: Key search, Grade (SLM), Generate (SLM)
        [
            actions.ACTION_RET_KEY, 
            actions.ACTION_GRD_SLM,
            actions.ACTION_GEN_SLM
        ],

        # Strategy 5.2: Vector search, Grade (LLM), Generate (LLM)
        [
            actions.ACTION_RET_VEC, 
            actions.ACTION_GRD_LLM,
            actions.ACTION_GEN_LLM
        ],
        # Strategy 6.1: Decompose, Retrieve, Grade (SLM), Answer (SLM) for each subtask
        [
            actions.ACTION_DEC_SLM,
            (actions.ACTION_RET_KEY, actions.ACTION_GRD_SLM, actions.ACTION_GEN_SLM), 
            actions.ACTION_GEN_SLM,     # Final synthesis
        ],

        # Strategy 6.2: Decompose, Retrieve, Grade (LLM), Answer (LLM) for each subtask
        [
            actions.ACTION_DEC_SLM,
            (actions.ACTION_RET_VEC, actions.ACTION_GRD_LLM, actions.ACTION_GEN_LLM), 
            actions.ACTION_GEN_LLM,     # Final synthesis
        ],
        # Strategy 7.1: Decompose, then for each subtask: Rewrite -> Retrieve -> Grade -> Answer
        [
            actions.ACTION_DEC_SLM,
            (actions.ACTION_RWT_SLM, actions.ACTION_RET_KEY, actions.ACTION_GRD_SLM, actions.ACTION_GEN_SLM),
            actions.ACTION_GEN_LLM,     # Final synthesis (using heavy model to tie it all together)
        ],
        
        # Strategy 7.2: Same as 7.1 but using Vector search and LLM grading
        [
            actions.ACTION_DEC_SLM,
            (actions.ACTION_RWT_SLM, actions.ACTION_RET_VEC, actions.ACTION_GRD_LLM, actions.ACTION_GEN_SLM),
            actions.ACTION_GEN_LLM,     
        ],
        # Strategy 8.1: Search -> Grade -> Rewrite -> Search -> Grade -> Answer
        [
            actions.ACTION_RET_KEY,
            actions.ACTION_GRD_SLM,
            actions.ACTION_RWT_SLM,     # "Slate cleared"
            actions.ACTION_RET_VEC,     # Try a different search method
            actions.ACTION_GRD_SLM,
            actions.ACTION_GEN_LLM
        ],
    ]


# Sort strategies cheapest-first using calibrated (or default) action costs.
# This ensures WaterfallOracle always tries the lowest-energy path before
# escalating to more expensive ones.
# Re-run src/oracle/calibrator.py to refresh cost_table.json whenever the
# model or hardware changes.
WaterfallOracle.STRATEGIES.sort(key=strategy_cost)
logging.debug(
    "WaterfallOracle strategy order (cheapest → most expensive):\n"
    + "\n".join(
        f"  [{i}] {round(strategy_cost(s), 4)} J  {s}"
        for i, s in enumerate(WaterfallOracle.STRATEGIES)
    )
)