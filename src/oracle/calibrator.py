import statistics
import json
from codecarbon import EmissionsTracker
from src.agent import actions, workers

class EnergyCalibrator:
    def __init__(self, iterations: int = 5, output_path: str = "data/meta/cost_table.json"):
        self.iterations = iterations
        self.output_path = output_path
        self.cost_table = {}

        # Dummy data for calibration
        self.dummy_docs = [
            "The capital of France is Paris.",
            "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
            "Python is a high-level, general-purpose programming language.",
        ]
        self.dummy_query = "What is the capital of France?"
        self.dummy_context = "The capital of France is Paris."
        self.dummy_response = "Paris."

    def _get_args_for_action(self, action_id):
        from src.env.state import GreenState
        # Need a Dummy State object now that workers expect State
        state = GreenState(question=self.dummy_query)
        state.context = self.dummy_context
        # History string simulation
        history_str = "Action: 2 | Arg: 'capital France' | Obs: 'Found 1 docs...'"

        if action_id in [actions.ACTION_GEN_SLM, actions.ACTION_GEN_LLM]:
            return (state, history_str)
        elif action_id in [actions.ACTION_RET_KEY, actions.ACTION_RET_VEC]:
            # Ret logic in workers.py expects (state, history) -> returns Query String
            # But the actual ACTION involves the Retriever, which is in src.env.retriever.
            # The worker function `generate_search_query` just generates the arg.
            # To calibrate the *full* cost, we should probably run the generation + the retrieval?
            # The prompt implies mapping IDs to functions in actions.py.
            # But we moved logic. 
            # Ideally we measure the *Agent* cost (Generation) + *Env* cost (Retrieval).
            # For this calibration, let's measure the Worker generation function as a proxy for the node cost,
            # or if the prompt implies the cost includes the environment step.
            # "Oracle must charge the Real World Energy Cost... in the search queue".
            # Let's measure the Worker functions defined in actions.py (if we kept them) or the new worker methods.
            return (state, history_str)
        elif action_id == actions.ACTION_GRD_SLM:
            return (state, self.dummy_context) 
        elif action_id == actions.ACTION_GRD_LLM:
            return (state, self.dummy_context)
        elif action_id in [actions.ACTION_RWT_SLM, actions.ACTION_DEC_SLM, actions.ACTION_DEC_LLM]:
            return (state, history_str)
        return ()

    def run(self):
        print("Initializing resources for calibration...")
        # Note: We need to init the workers/models if not already
        # They are global in src.agent.workers, so importing them helps.
        
        print(f"Starting calibration with {self.iterations} iterations per action...")
        
        # We need to map Action IDs to the actual function calls we want to measure.
        # Since we refactored, we need to bind them manually here.
        
        for action_id, name in actions.ACTION_NAMES.items():
            if action_id == actions.ACTION_FAIL:
                self.cost_table[str(action_id)] = 0.0
                continue
                
            print(f"Calibrating {name}...")
            
            # Select function and args
            func = None
            args = self._get_args_for_action(action_id)
            
            if action_id == actions.ACTION_GEN_SLM:
                func = lambda s, h: workers.generate_answer(s, h, use_llm=False)
            elif action_id == actions.ACTION_GEN_LLM:
                func = lambda s, h: workers.generate_answer(s, h, use_llm=True)
            elif action_id == actions.ACTION_RET_KEY:
                # Measure Generation of query 
                # (Retrieval itself is "Ephemeral" and fast in this sim, but we might want to price the generation)
                func = lambda s, h: workers.generate_search_query(s, h)
            elif action_id == actions.ACTION_RET_VEC:
                 func = lambda s, h: workers.generate_search_query(s, h)
            elif action_id == actions.ACTION_GRD_SLM:
                func = lambda s, d: workers.generate_grade(s, d, use_llm=False)
            elif action_id == actions.ACTION_GRD_LLM:
                 func = lambda s, d: workers.generate_grade(s, d, use_llm=True)
            elif action_id == actions.ACTION_RWT_SLM:
                func = lambda s, h: workers.generate_rewrite(s, h)
            elif action_id == actions.ACTION_DEC_SLM:
                func = lambda s, h: workers.generate_plan(s, h, use_llm=False)
            elif action_id == actions.ACTION_DEC_LLM:
                func = lambda s, h: workers.generate_plan(s, h, use_llm=True)

            if not func:
                print(f"Skipping {name} (No function mapped)")
                continue

            energies = []
            
            for i in range(self.iterations):
                tracker = EmissionsTracker(output_dir="/tmp", log_level="error", measure_power_secs=0.1)
                tracker.start()
                try:
                    func(*args)
                except Exception as e:
                    print(f"Error running {name}: {e}")
                finally:
                    tracker.stop()
                    energy_kwh = tracker.final_emissions_data.energy_consumed
                    energy_joules = energy_kwh * 3_600_000
                    energies.append(energy_joules)
                    
            avg_joules = statistics.mean(energies) if energies else 0.0
            print(f"  -> {avg_joules:.4f} Joules (avg)")
            self.cost_table[str(action_id)] = avg_joules

        self.save()

    def save(self):
        with open(self.output_path, "w") as f:
            json.dump(self.cost_table, f, indent=2)
        print(f"Calibration complete. Cost table saved to {self.output_path}")
