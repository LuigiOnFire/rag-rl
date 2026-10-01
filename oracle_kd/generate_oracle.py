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
from src.data.loader import MixedStreamer
from src.env.engine import GreenEngine
from src.env.retriever import EphemeralRetriever, GlobalRetriever
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


AVG_DECOMPOSE_SUBQUERIES = 3


def strategy_cost(strategy: list, cost_table: Dict[str, float]) -> float:
    total = 0.0
    for entry in strategy:
        if isinstance(entry, tuple):
            total += AVG_DECOMPOSE_SUBQUERIES * sum(get_action_cost(cost_table, a) for a in entry)
        else:
            total += get_action_cost(cost_table, entry)
    return total


# =========================================================================
# 7-Route Minimal Representative Trajectories
# =========================================================================

# Route 0: Direct SLM
def traj_direct_slm() -> List:
    return [actions.ACTION_GEN_SLM]

# Route 1: Direct LLM
def traj_direct_llm() -> List:
    return [actions.ACTION_GEN_LLM]

# Route 2: Keyword -> SLM
def traj_key_then_slm() -> List:
    return [actions.ACTION_RET_KEY, actions.ACTION_GEN_SLM]

# Route 3: Vector Search -> LLM
def traj_vec_then_llm() -> List:
    return [actions.ACTION_RET_VEC, actions.ACTION_GEN_LLM]

# Route 4: SLM Reason -> Vector Search -> LLM
def traj_reason_vec_llm() -> List:
    return [actions.ACTION_RSN_SLM, actions.ACTION_RET_VEC, actions.ACTION_GEN_LLM]

# Route 5: Decompose -> Sub-retrieval w/ LLM Reason & Sub-Answer
def traj_decompose_retrieve_reason() -> List:
    return [
        actions.ACTION_DEC_LLM,
        (actions.ACTION_RET_VEC, actions.ACTION_RSN_LLM, actions.ACTION_GEN_LLM),
        actions.ACTION_GEN_LLM,
    ]

# Route 6: Heavy Joint Reason & Decompose -> Full Loop
def traj_heavy_decompose_retrieve_reason() -> List:
    return [
        actions.ACTION_DEC_RSN,
        (actions.ACTION_RET_VEC, actions.ACTION_RSN_LLM, actions.ACTION_GEN_LLM),
        actions.ACTION_GEN_LLM,
    ]


def build_trajectories(cost_table: Dict[str, float], auto_sort: bool = True) -> List[Dict[str, object]]:
    trajectories = [
        {"name": "direct_slm", "fn": traj_direct_slm},
        {"name": "direct_llm", "fn": traj_direct_llm},
        {"name": "key_then_slm", "fn": traj_key_then_slm},
        {"name": "vec_then_llm", "fn": traj_vec_then_llm},
        {"name": "reason_vec_llm", "fn": traj_reason_vec_llm},
        {"name": "decompose_retrieve_reason", "fn": traj_decompose_retrieve_reason},
        {"name": "heavy_decompose_retrieve_reason", "fn": traj_heavy_decompose_retrieve_reason},
    ]

    if auto_sort:
        # Dynamically sort strictly by strategy cost to prevent cost-inversion bugs
        trajectories = sorted(trajectories, key=lambda t: strategy_cost(t["fn"](), cost_table))
        logging.info("Trajectories sorted by strategy cost:")
        for idx, t in enumerate(trajectories):
            logging.info("  [%d] %-32s : %.2f J", idx, t["name"], strategy_cost(t["fn"](), cost_table))
    else:
        # Hard assertion check
        costs = [strategy_cost(t["fn"](), cost_table) for t in trajectories]
        for i in range(len(costs) - 1):
            if costs[i] > costs[i + 1]:
                raise AssertionError(
                    f"Trajectory list is not cost-ordered: {trajectories[i]['name']} ({costs[i]:.2f} J) > "
                    f"{trajectories[i+1]['name']} ({costs[i+1]:.2f} J). Pass --auto-sort-trajectories to fix."
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
                    current_state = engine.step(current_state, sub_action, argument=None)
                    if current_state["status"] in ("SOLVED", "FAILED"):
                        break
                loop_safety_counter += 1
                if current_state["status"] in ("SOLVED", "FAILED"):
                    break
        else:
            current_state = engine.step(current_state, entry, argument=None)

        if current_state["status"] in ("SOLVED", "FAILED"):
            break

    return current_state


def init_csv(output_path: str) -> None:
    if os.path.exists(output_path):
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "optimal_trajectory_id", "joules_spent", "is_correct"],
        )
        writer.writeheader()


def append_jsonl(output_path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def append_rows(output_path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "optimal_trajectory_id", "joules_spent", "is_correct"],
        )
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cascading oracle labels.")
    parser.add_argument("--datasets", nargs="+", default=["hotpot"], help="Dataset names.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="hotpotqa",
        choices=["hotpotqa", "squad", "nq"],
        help="High-level dataset selector (hotpotqa, squad, or natural questions).",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="hnsw",
        choices=["hnsw", "ivf", "flat"],
        help="FAISS dense index topology to load.",
    )
    parser.add_argument("--limit", type=int, default=500, help="Smoke test sample limit.")
    parser.add_argument("--setting", default="fullwiki", help="Dataset setting.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--output", default="data/oracle/oracle_training_data.csv")
    parser.add_argument("--history-output", default="data/oracle/oracle_trajectory_history.jsonl")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--offset", type=int, default=0, help="Number of samples to skip before starting.")
    parser.add_argument(
        "--execution-mode",
        type=str,
        default="all_routes",
        choices=["first_success", "all_routes"],
        help="Route execution mode: stop at first correct route or run all routes and log each attempt.",
    )
    parser.add_argument(
        "--auto-sort-trajectories",
        action="store_true",
        default=True,
        help="Sort candidate trajectories strictly by cost using cost_table.json to prevent inversions.",
    )
    return parser.parse_args()


def main() -> None:
    # 1. Force root logger to INFO and suppress noisy third-party loggers
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )
    for noisy in ["fsspec", "urllib3", "filelock", "datasets", "transformers", "httpcore", "httpx"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    args = parse_args()
    
    default_output = "data/oracle/oracle_training_data.csv"
    if args.output == default_output:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"data/oracle/oracle_training_data_{args.dataset_name}_{run_id}.csv"

    default_history_output = "data/oracle/oracle_trajectory_history.jsonl"
    if args.history_output == default_history_output:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        args.history_output = f"data/oracle/oracle_trajectory_history_{args.dataset_name}_{run_id}.jsonl"

    trace_run_id = time.strftime("%Y%m%d_%H%M%S")
    trace_dir = f"data/oracle/oracle_{trace_run_id}_query_traces"
    os.makedirs(trace_dir, exist_ok=True)
    logging.info("Worker trace logs will be written to %s", trace_dir)

    cost_table = load_cost_table()
    trajectories = build_trajectories(cost_table, auto_sort=args.auto_sort_trajectories)
    judge = SoftJudge()

    # Map dataset selection
    if args.dataset_name == "nq":
        dataset_names = ["nq"]
    elif args.dataset_name == "squad":
        dataset_names = ["squad"]
    else:
        dataset_names = ["hotpot"]
        
    dataset_configs = {
        name: {"setting": args.setting, "split": args.split} for name in dataset_names
    }
    logging.info("Using datasets: %s", ", ".join(dataset_names))

    streamer = MixedStreamer(
        dataset_names=dataset_names,
        limit=args.limit + args.offset,        
        shuffle=args.shuffle,
        configs=dataset_configs,
    )

    logging.info(
        "Streaming %s examples (starting at offset %s) of %s available from %s",
        streamer.n_limit,
        args.offset,
        streamer.total_available,
        ", ".join(dataset_names),    
    )

    init_csv(args.output)

    processed_count = 0
    for idx, sample in enumerate(streamer.stream()):
        if idx < args.offset:
            if (idx + 1) % 1000 == 0:
                logging.info("Skipping offset rows... (%d/%d)", idx + 1, args.offset)
            continue

        if processed_count >= args.limit:
            logging.info("Reached limit of %d generated samples. Stopping.", args.limit)
            break

        question = sample["question"]
        ground_truth = sample.get("ground_truth") or sample.get("answer", "")
        corpus = sample.get("corpus", [])

        query_log_path = os.path.join(trace_dir, f"q_{processed_count:06d}.log")
        workers.configure_worker_logging(query_log_path)

        if args.setting == "distractor":
            if not corpus:
                logging.warning("Skipping sample with empty distractor corpus: %s", question)
                continue
            retriever = EphemeralRetriever(documents=corpus)

        elif args.setting == "fullwiki":
            corpus_type = "dpr_wiki" if args.dataset_name in ("squad", "nq") else "fullwiki"
            retriever = GlobalRetriever.get_instance(
                corpus_type=corpus_type,
                index_type=args.index_type
            )

        engine = GreenEngine(retriever=retriever)

        chosen_id = None
        joules_spent = 0.0
        is_correct = False
        last_state = None
        attempt_records = []

        # MAIN CASCADING SEARCH LOOP
        for traj_idx, traj in enumerate(trajectories):
            start_state = create_initial_state(question)
            strategy = traj["fn"]()

            t0 = time.perf_counter()
            final_state = run_strategy(engine, start_state, strategy)
            t1 = time.perf_counter()
            trajectory_duration_sec = t1 - t0

            last_state = final_state

            final_answer = final_state.get("answer") or ""
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
                    is_correct = True

                if args.execution_mode == "first_success":
                    break

        if chosen_id is None:
            chosen_id = len(trajectories) - 1
            if last_state is not None:
                joules_spent = float(last_state.get("total_joules", 0.0))

        append_rows(
            args.output,
            [
                {
                    "question": question,
                    "optimal_trajectory_id": chosen_id,
                    "joules_spent": joules_spent,
                    "is_correct": is_correct,
                }
            ],
        )
        append_jsonl(
            args.history_output,
            [
                {
                    "question": question,
                    "source": args.dataset_name, 
                    "optimal_trajectory_id": chosen_id,
                    "joules_spent": joules_spent,
                    "is_correct": is_correct,
                    "execution_mode": args.execution_mode,
                    "attempts": attempt_records,
                }
            ],
        )

        processed_count += 1
        logging.info(
            "[%d/%d] Q: %s... -> Winner: %s (Correct: %s, Joules: %.1f J)",
            processed_count,
            args.limit,
            question[:50],
            trajectories[chosen_id]["name"],
            is_correct,
            joules_spent
        )

    logging.info("Smoke test complete! Data written to %s", args.output)


if __name__ == "__main__":
    main()
