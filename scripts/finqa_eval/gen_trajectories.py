import argparse
import csv
import json
import logging
import os
import sys
import time
from typing import Callable, Dict, List, Tuple

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.agent import actions
from src.agent import workers
# Ensure FinQAStreamer is imported from wherever you saved it
from src.data.finqa import FinQAStreamer 
from src.env.engine import GreenEngine
from src.env.retriever import EphemeralRetriever
from src.env.state import create_initial_state, get_active_subquery, GreenState
from src.oracle.judge import SoftJudge

TrajectoryFn = Callable[[], List]

def load_cost_table(cost_table_path: str = "data/meta/cost_table.json") -> Dict[str, float]:
    try:
        with open(cost_table_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning("%s not found. Using default costs (1.0).", cost_table_path)
        return {}

def get_action_cost(cost_table: Dict[str, float], action_id: int) -> float:
    return float(cost_table.get(str(action_id), 1.0))

# Keep this aligned with the calibrator assumption used elsewhere.
AVG_DECOMPOSE_SUBQUERIES = 3

def strategy_cost(strategy: list, cost_table: Dict[str, float]) -> float:
    total = 0.0
    for entry in strategy:
        if isinstance(entry, tuple):
            total += AVG_DECOMPOSE_SUBQUERIES * sum(get_action_cost(cost_table, a) for a in entry)
        else:
            total += get_action_cost(cost_table, entry)
    return total

# --- Trajectory Definitions ---
def traj_key_then_slm() -> List: return [actions.ACTION_RET_KEY, actions.ACTION_GEN_SLM]
def traj_vec_then_llm() -> List: return [actions.ACTION_RET_VEC, actions.ACTION_GEN_LLM]
def traj_reason_vec_llm() -> List: return [actions.ACTION_RSN_SLM, actions.ACTION_RET_VEC, actions.ACTION_GEN_LLM]
def traj_search_reason_iterate() -> List:
    return [actions.ACTION_RET_KEY, actions.ACTION_RSN_SLM, actions.ACTION_RET_VEC, actions.ACTION_RSN_SLM, actions.ACTION_GEN_LLM]
def traj_decompose_key_slm() -> List:
    return [actions.ACTION_DEC_LLM, (actions.ACTION_RET_KEY, actions.ACTION_GEN_SLM), actions.ACTION_GEN_SLM]
def traj_decompose_retreive_reason() -> List:
    return [actions.ACTION_DEC_LLM, (actions.ACTION_RET_VEC, actions.ACTION_RSN_SLM, actions.ACTION_GEN_SLM), actions.ACTION_GEN_LLM]
def traj_heavy_decompose_retreive_reason() -> List:
    return [actions.ACTION_DEC_RSN, (actions.ACTION_RET_VEC, actions.ACTION_RSN_SLM, actions.ACTION_GEN_LLM), actions.ACTION_GEN_LLM]

def build_trajectories() -> List[Dict[str, object]]:
    trajectories = [
        {"name": "key_then_slm", "fn": traj_key_then_slm},
        {"name": "vec_then_llm", "fn": traj_vec_then_llm},
        {"name": "reason_vec_llm", "fn": traj_reason_vec_llm},
        {"name": "search_reason_iterate", "fn": traj_search_reason_iterate},
        {"name": "decompose_key_slm", "fn": traj_decompose_key_slm},        
        {"name": "decompose_retreive_reason", "fn": traj_decompose_retreive_reason},
        {"name": "heavy_decompose_retreive_reason", "fn": traj_heavy_decompose_retreive_reason}
    ]

    cost_table = load_cost_table()
    costs = [strategy_cost(t["fn"](), cost_table) for t in trajectories]
    
    # Validation check: Ensure trajectories are sorted cheapest -> most expensive
    for i in range(len(costs) - 1):
        if costs[i] > costs[i + 1]:
            raise AssertionError(
                "Trajectory list is not cost-ordered. "
                f"{trajectories[i]['name']} ({costs[i]:.4f}) > "
                f"{trajectories[i + 1]['name']} ({costs[i + 1]:.4f})."
            )
    return trajectories


def run_strategy(engine: GreenEngine, start_state: GreenState, strategy: List) -> GreenState:
    current_state = start_state
    for entry in strategy:
        if isinstance(entry, tuple):
            repeat_actions = entry
            loop_safety_counter = 0  
            while get_active_subquery(current_state) is not None and loop_safety_counter < 20: 
                for sub_action in repeat_actions:
                    current_state = engine.step(current_state, sub_action, argument=None, task_type="fin")
                    if current_state["status"] in ("SOLVED", "FAILED"):
                        break
                loop_safety_counter += 1
                if current_state["status"] in ("SOLVED", "FAILED"):
                    break
        else:
            current_state = engine.step(current_state, entry, argument=None, task_type="fin")

        if current_state["status"] in ("SOLVED", "FAILED"):
            break

    return current_state


def init_csv(output_path: str) -> None:
    if os.path.exists(output_path): return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "optimal_trajectory_id", "joules_spent", "is_correct"])
        writer.writeheader()

def append_jsonl(output_path: str, rows: List[Dict[str, object]]) -> None:
    if not rows: return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a") as f:
        for row in rows: f.write(json.dumps(row) + "\n")

def append_rows(output_path: str, rows: List[Dict[str, object]]) -> None:
    if not rows: return
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "optimal_trajectory_id", "joules_spent", "is_correct"])
        writer.writerows(rows)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cascading oracle labels for FinQA.")
    parser.add_argument("--limit", type=int, default=1000, help="Samples to evaluate.") 
    parser.add_argument("--split", default="train", help="FinQA split (train/test/dev).")
    parser.add_argument("--output", default="data/oracle/finqa_oracle_training_data.csv")
    parser.add_argument("--history-output", default="data/oracle/finqa_oracle_trajectory_history.jsonl")
    parser.add_argument("--offset", type=int, default=0, help="Samples to skip before starting.")
    parser.add_argument(
        "--execution-mode",
        type=str,
        default="first_success",
        choices=["first_success", "all_routes"],
        help="Stop at first correct route or run all routes.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    args = parse_args()
    
    # Auto-generate unique filenames if using defaults
    run_id = time.strftime("%Y%m%d_%H%M%S")
    if args.output == "data/oracle/finqa_oracle_training_data.csv":
        args.output = f"data/oracle/finqa_oracle_training_data_{run_id}.csv"
    if args.history_output == "data/oracle/finqa_oracle_trajectory_history.jsonl":
        args.history_output = f"data/oracle/finqa_oracle_trajectory_history_{run_id}.jsonl"

    trace_dir = f"data/oracle/finqa_oracle_{run_id}_query_traces"
    os.makedirs(trace_dir, exist_ok=True)
    logging.info("Worker trace logs will be written to %s", trace_dir)

    trajectories = build_trajectories()
    cost_table = load_cost_table()
    judge = SoftJudge()

    streamer = FinQAStreamer(split=args.split, limit=args.limit + args.offset)
    init_csv(args.output)

    total_samples = streamer.total_size if hasattr(streamer, 'total_size') else "Unknown"
    logging.info("Streaming FinQA examples (offset %s, limit %s)", args.offset, args.limit)

    processed_count = 0
    for idx, sample in enumerate(streamer.stream()):
        if idx < args.offset:
            if (idx + 1) % 100 == 0:
                logging.info(f"Skipping offset rows... ({idx + 1}/{args.offset})")
            continue

        if processed_count >= args.limit:
            logging.info(f"Reached limit of {args.limit} generated samples. Stopping.")
            break

        question = sample["question"]
        ground_truth = sample.get("answer", "")
        
        query_log_path = os.path.join(trace_dir, f"q_{processed_count:06d}.log")
        workers.configure_worker_logging(query_log_path)

        # 1. Initialize Retriever and Engine PER QUESTION for FinQA
        doc_strings = [f"{doc['title']}:\n{doc['text']}" for doc in sample["corpus"]]
        retriever = EphemeralRetriever(documents=doc_strings)
        engine = GreenEngine(retriever=retriever)

        chosen_id = None
        joules_spent = 0.0
        is_correct = False
        last_state = None
        best_correct_state = None
        attempt_records = []

        # 2. Test All Trajectories
        for traj_idx, traj in enumerate(trajectories):
            # Create fresh state for this trajectory attempt
            start_state = create_initial_state(question)
            start_state["context"] = sample.get("raw_context", "") # Inject raw text just in case
            
            strategy = traj["fn"]()

            t0 = time.perf_counter()
            final_state = run_strategy(engine, start_state, strategy)
            trajectory_duration_sec = time.perf_counter() - t0
            last_state = final_state

            final_answer = final_state.get("answer") or ""
            
            # TODO: Replace with judge.judge_numeric once calculator is integrated
            judged_correct, _ = judge.judge(final_answer, ground_truth, question)
            
            measured_joules = float(final_state.get("total_joules", 0.0))

            attempt_records.append({
                "trajectory_id": traj_idx,
                "trajectory_name": traj["name"],
                "estimated_cost": float(strategy_cost(strategy, cost_table)),
                "measured_joules": measured_joules,
                "duration_seconds": trajectory_duration_sec,
                "is_correct": bool(judged_correct),
                "status": final_state.get("status", ""),
                "history": final_state.get("history", []) 
            })

            if judged_correct:
                if chosen_id is None:
                    chosen_id = traj_idx
                    joules_spent = measured_joules
                    best_correct_state = final_state
                    is_correct = True

                if args.execution_mode == "first_success":
                    break

        if chosen_id is None:
            chosen_id = len(trajectories) - 1
            if last_state is not None:
                joules_spent = float(last_state.get("total_joules", 0.0))

        append_rows(
            args.output,
            [{"question": question, "optimal_trajectory_id": chosen_id, "joules_spent": joules_spent, "is_correct": is_correct}]
        )
        
        append_jsonl(
            args.history_output,
            [{
                "question": question,
                "source": "finqa", 
                "optimal_trajectory_id": chosen_id,
                "joules_spent": joules_spent,
                "is_correct": is_correct,
                "execution_mode": args.execution_mode,
                "attempts": attempt_records,
            }]
        )

        processed_count += 1
        if processed_count % 10 == 0:
            logging.info("Processed %s/%s", processed_count, args.limit)

    logging.info("Done. Output saved to %s", args.output)

if __name__ == "__main__":
    main()