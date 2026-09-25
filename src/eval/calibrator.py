import statistics
import json
import os
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
        print(f"Starting calibration with {self.iterations} iterations per action...")

        # 1. Initialize the global tracker EXACTLY ONCE to prevent [Errno 24] file leaks
        tracker = EmissionsTracker(output_dir="/tmp", log_level="error", measure_power_secs=0.1)
        tracker.start()

        try:
            for action_id in actions.ALL_ACTION_IDS:
                name = actions.get_action_name(action_id)
                if action_id == actions.ACTION_FAIL:
                    self.cost_table[str(action_id)] = 0.0
                    continue

                print(f"Calibrating {name}...")

                # Initialize the retriever and GreenEngine
                retriever = EphemeralRetriever(documents=self.dummy_docs)
                self.engine = GreenEngine(retriever=retriever)

                func = lambda s: self.engine.step(s, action_id, argument=None)

                if not func:
                    print(f"Skipping {name} (No function mapped)")
                    continue

                energies = []

                for i in range(self.iterations):
                    task_name = f"task_{action_id}_{i}"
                    
                    # 2. Start a lightweight sub-task for just this iteration
                    tracker.start_task(task_name)
                    
                    try:
                        state = create_initial_state(self.dummy_query)
                        func(state)
                    except Exception as e:
                        print(f"Error running {name}: {e}")
                    finally:
                        # 3. Stop the sub-task and extract the isolated energy delta
                        task = tracker.stop_task(task_name)
                        
                        # Fallback logic to support different versions of CodeCarbon
                        if task and hasattr(task, 'emissions_data'):
                            energy_kwh = task.emissions_data.energy_consumed
                        elif task and hasattr(task, 'energy_consumed'):
                            energy_kwh = task.energy_consumed
                        else:
                            energy_kwh = 0.0
                            
                        energy_joules = energy_kwh * 3_600_000
                        energies.append(energy_joules)   
                        
                avg_joules = statistics.mean(energies) if energies else 0.0
                print(f"  -> {avg_joules:.4f} Joules (avg)")
                self.cost_table[str(action_id)] = avg_joules

        finally:
            # 4. Shutdown the global tracker perfectly at the end of all loops
            tracker.stop()
            self.save()

    def save(self):
        project_root = os.getcwd()
        print(f"We've registered the project root as {project_root}")
        directory = os.path.dirname(self.output_path)

        if directory:
            os.makedirs(directory, exist_ok=True)  # <-- Fixed the typo here too!
        with open(self.output_path, "w") as f:
            json.dump(self.cost_table, f, indent=2)
        print(f"Calibration complete. Cost table saved to {self.output_path}")
