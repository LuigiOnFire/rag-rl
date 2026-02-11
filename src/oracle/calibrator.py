import statistics
import json
from codecarbon import EmissionsTracker
from src.agent import actions, workers
from src.env.state import GreenState, create_initial_state
from src.env.retriever import EphemeralRetriever
from src.env.engine import GreenEngine


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

    def run(self):
        print("Initializing resources for calibration...")
        # Note: We need to init the workers/models if not already
        # They are global in src.agent.workers, so importing them helps.
        
        print(f"Starting calibration with {self.iterations} iterations per action...")
        
        # We need to map Action IDs to the actual function calls we want to measure.
        # Since we refactored, we need to bind them manually here.
        
        for action_id in actions.ALL_ACTION_IDS:
            name = actions.get_action_name(action_id)
            if action_id == actions.ACTION_FAIL:
                self.cost_table[str(action_id)] = 0.0
                continue

            print(f"Calibrating {name}...")

            # Initialize the retrievre and GreenEngine
            retriever = EphemeralRetriever(documents=self.dummy_docs)
            self.engine = GreenEngine(retriever=retriever)
            
            func = lambda s: self.engine.step(s, action_id, argument=None)

            if not func:
                print(f"Skipping {name} (No function mapped)")
                continue

            energies = []
            
            for i in range(self.iterations):
                tracker = EmissionsTracker(output_dir="/tmp", log_level="error", measure_power_secs=0.1)
                tracker.start()
                try:
                    state = create_initial_state(self.dummy_query)
                    func(state)
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
