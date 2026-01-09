import heapq
import json
import copy
from typing import List, Optional, Dict
from src.agent import actions, workers
from src.env.state import GreenState, GreenStep
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
        valid = []
        last = state.last_action
        
        # Start of Turn (or if last action indicates a reset/new turn logic, but simplistic view: Start if history empty)
        if last is None:
             # {RET, DEC, GEN_SLM}
             return [actions.ACTION_RET_KEY, actions.ACTION_RET_VEC, 
                     actions.ACTION_DEC_SLM, actions.ACTION_DEC_LLM, 
                     actions.ACTION_GEN_SLM, actions.ACTION_GEN_LLM]

        # After RET (Retrieval): Valid → {GRD, GEN}
        if last in [actions.ACTION_RET_KEY, actions.ACTION_RET_VEC]:
            return [actions.ACTION_GRD_SLM, actions.ACTION_GRD_LLM, 
                    actions.ACTION_GEN_SLM, actions.ACTION_GEN_LLM]

        # After GRD (Grading)
        if last in [actions.ACTION_GRD_SLM, actions.ACTION_GRD_LLM]:
            # Check Observation of last step
            obs = state.last_observation.lower()
            if "relevant" in obs and "irrelevant" not in obs:
                 # Grade="Good": Valid → {GEN, RWT}
                 return [actions.ACTION_GEN_SLM, actions.ACTION_GEN_LLM, actions.ACTION_RWT_SLM]
            else:
                 # Grade="Bad": Valid → {RET_VEC, DEC}
                 return [actions.ACTION_RET_VEC, actions.ACTION_DEC_SLM, actions.ACTION_DEC_LLM]

        # After RWT (Rewrite): Valid → {RET}
        if last == actions.ACTION_RWT_SLM:
            return [actions.ACTION_RET_KEY, actions.ACTION_RET_VEC] # Should we allow Key? Prompt says {RET}

        # After DEC (Decompose): Valid → {RET}
        if last in [actions.ACTION_DEC_SLM, actions.ACTION_DEC_LLM]:
            return [actions.ACTION_RET_KEY, actions.ACTION_RET_VEC]

        return []

    def solve(self, start_state: GreenState, max_depth=10) -> Optional[GreenState]:
        # Priority Queue: (TotalCost, State)
        # Note: heapq compares tuples element by element. State needs comparison or wrapped.
        # We'll use id(state) as tiebreaker or wrap it.
        
        pq = [(0.0, 0, start_state)] # (cost, tiebreaker, state)
        tiebreaker = 0
        
        visited_hashes = set()
        
        best_cost = float('inf')
        solution = None

        while pq:
            cost, _, current_state = heapq.heappop(pq)
            
            # Pruning
            if cost >= best_cost:
                continue
            
            # Terminal Check (GEN)
            last_action = current_state.last_action
            if last_action in [actions.ACTION_GEN_SLM, actions.ACTION_GEN_LLM]:
                if self.judge.judge(current_state.last_observation, current_state.ground_truth):
                    # BFS/Dijkstra guarantees the first valid solution found is the cheapest.
                    return current_state
            
            if current_state.depth >= max_depth:
                continue

            # Expand
            valid_actions = self.get_valid_actions(current_state)
            
            for action_id in valid_actions:
                # Execute Action Logic
                try:
                    # Create deep copy for new branch
                    new_state = copy.deepcopy(current_state)
                    
                    # Generate Argument & Execute
                    hist_str = workers._format_history(current_state)
                    argument = ""
                    observation = ""
                    
                    if action_id in [actions.ACTION_RET_KEY, actions.ACTION_RET_VEC]:
                        # Generate Query
                        argument = workers.generate_search_query(new_state, hist_str)
                        # Execute Search
                        if action_id == actions.ACTION_RET_KEY:
                            docs = self.retriever.search_bm25(argument)
                        else:
                            docs = self.retriever.search_dense(argument)
                        observation = f"Found {len(docs)} docs: " + " ".join(docs)[:200] + "..." # Truncate for state string?
                        # Store docs in context?
                        new_state.context += "\n" + "\n".join(docs)
                        
                    elif action_id in [actions.ACTION_GRD_SLM, actions.ACTION_GRD_LLM]:
                        # Grade last retrieved docs (assumed in new_state.context or we fetch last step?)
                        # Logic: Grade needs specific doc. For simplicity, we grade the *last added context*.
                        # or just pass the full context?
                        # "History + Last Retrieved Docs"
                        # Recover last docs from history or context?
                        # Simplification: use context.
                        argument = "Check Relevance" # Implicit
                        observation = workers.generate_grade(new_state, new_state.context[-1000:], 
                                                             use_llm=(action_id==actions.ACTION_GRD_LLM))
                        
                    elif action_id == actions.ACTION_RWT_SLM:
                        argument = workers.generate_rewrite(new_state, hist_str)
                        observation = f"Query rewritten to: {argument}"
                        
                    elif action_id in [actions.ACTION_DEC_SLM, actions.ACTION_DEC_LLM]:
                        argument = workers.generate_plan(new_state, hist_str, 
                                                         use_llm=(action_id==actions.ACTION_DEC_LLM))
                        observation = f"Plan: {argument}"
                        
                    elif action_id in [actions.ACTION_GEN_SLM, actions.ACTION_GEN_LLM]:
                        argument = "Generate Answer"
                        observation = workers.generate_answer(new_state, hist_str, 
                                                              use_llm=(action_id==actions.ACTION_GEN_LLM))

                    # Update State
                    step: GreenStep = {
                        "action_id": action_id,
                        "argument": argument,
                        "observation": observation,
                        "cost": get_cost(action_id)
                    }
                    new_state.add_step(step)
                    
                    tiebreaker += 1
                    heapq.heappush(pq, (cost + step['cost'], tiebreaker, new_state))
                    
                except Exception as e:
                    print(f"Error executing action {action_id}: {e}")
                    continue
                    
        return solution
