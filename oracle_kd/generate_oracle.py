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


def traj_direct_llm() -> List:
    return [actions.ACTION_GEN_LLM]

def traj_key_then_slm() -> List:
    return [actions.ACTION_RET_KEY, actions.ACTION_GEN_SLM]

def traj_vec_then_llm() -> List:
    return [actions.ACTION_RET_VEC, actions.ACTION_GEN_LLM]

def traj_reason_vec_llm() -> List:
    return [actions.ACTION_RSN_SLM, actions.ACTION_RET_VEC, actions.ACTION_GEN_LLM]

def traj_search_reason_iterate() -> List:
    return [
        actions.ACTION_RET_KEY,
        actions.ACTION_RSN_SLM,
        actions.ACTION_RET_VEC,     
        actions.ACTION_RSN_SLM,
        actions.ACTION_GEN_LLM,
    ]

def traj_decompose_key_slm() -> List:
    return [
        actions.ACTION_DEC_LLM,
        (actions.ACTION_RET_KEY, actions.ACTION_GEN_SLM),
        actions.ACTION_GEN_SLM,
    ]

def traj_decompose_retreive_reason() -> List:
    return [
        actions.ACTION_DEC_LLM,
        (actions.ACTION_RET_VEC, actions.ACTION_RSN_SLM, actions.ACTION_GEN_SLM),
        actions.ACTION_GEN_LLM,
    ]

def traj_heavy_decompose_retreive_reason() -> List:
    return [
        actions.ACTION_DEC_RSN,
        (actions.ACTION_RET_VEC, actions.ACTION_RSN_SLM, actions.ACTION_GEN_LLM),
        actions.ACTION_GEN_LLM,
    ]

def build_trajectories() -> List[Dict[str, object]]:
    trajectories = [
        {"name": "direct_llm", "fn": traj_direct_llm},
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
    for i in range(len(costs) - 1):
        if costs[i] > costs[i + 1]:

            raise AssertionError(
                "Trajectory list is not cost-ordered. "
                f"{trajectories[i]['name']} ({costs[i]:.4f}) > "
                f"{trajectories[i + 1]['name']} ({costs[i + 1]:.4f})."
                f"Order should be: {[t['name'] + ': ' + str(strategy_cost(t['fn'](), cost_table)) for t in sorted(trajectories, key=lambda x: strategy_cost(x['fn'](), cost_table))]}"
            )

    return trajectories


def run_strategy(engine: GreenEngine, start_state: GreenState, strategy: List) -> GreenState:
    current_state = start_state

    for entry in strategy:
        if isinstance(entry, tuple):
            repeat_actions = entry
            loop_safety_counter = 0  
            while get_active_subquery(current_state) is not None and loop_safety_counter < 20:  # Safety to prevent infinite loops
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
    parser.add_argument("--limit", type=int, default=11000, help="Samples per dataset.") # Recalculated for about 60 hours but with fullwiki it will be slower
    parser.add_argument("--setting", default="fullwiki", help="Dataset setting.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--output", default="data/oracle/oracle_training_data.csv")
    parser.add_argument("--history-output", default="data/oracle/oracle_trajectory_history.jsonl")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--offset", type=int, default=0, help="Number of samples to skip before starting.")
    parser.add_argument(
        "--execution-mode",
        type=str,
        default="first_success",
        choices=["first_success", "all_routes"],
        help="Route execution mode: stop at first correct route or run all routes and log each attempt.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Silence verbose HTTP debug logs from downstream clients.
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

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

    trajectories = build_trajectories()
    cost_table = load_cost_table()
    judge = SoftJudge()

    dataset_configs = {
        name: {"setting": args.setting, "split": args.split} for name in args.datasets
    }

    # Determine dataset names based on --dataset-name arg
# Ensure the streamer loads the correct dataset based on the high-level arg
    if args.dataset_name == "nq":
        dataset_names = ["nq"]
    elif args.dataset_name == "squad":
        dataset_names = ["squad"]
    else:
        dataset_names = ["hotpot"]
        
    dataset_configs = {
        name: {"setting": args.setting, "split": args.split} for name in dataset_names
    }
    print("Using the following datasets: {}".format(", ".join(dataset_names)))

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

    total = streamer.n_limit
    processed_count = 0  # how many we've actually done, which may be less than idx due to offset and skips
    for idx, sample in enumerate(streamer.stream()):
        # skip for offset
        if idx < args.offset:
            if (idx + 1) % 1000 == 0:
                logging.info(f"Skipping offset rows... ({idx + 1}/{args.offset})")
            continue

        # stop for the limit 
        if processed_count >= args.limit:
            logging.info(f"Reached limit of {args.limit} generated samples. Stopping.")
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
            # Build a tiny 10-paragraph retriever
            retriever = EphemeralRetriever(documents=corpus)

        elif args.setting == "fullwiki":
            # Select retriever corpus based on dataset
            # SQuAD and NQ use DPR Wikipedia (squad_wiki), others use default Wikipedia (fullwiki)
            if args.dataset_name == "squad" or args.dataset_name == "nq":
                retriever = GlobalRetriever.get_instance(corpus_type="dpr_wiki")
            else:
                retriever = GlobalRetriever.get_instance(corpus_type="fullwiki")

        engine = GreenEngine(retriever=retriever)

        chosen_id = None
        joules_spent = 0.0
        is_correct = False
        last_state = None
        best_correct_state = None
        attempt_records = []

        # THIS IS THE MAIN LOOP
        for traj_idx, traj in enumerate(trajectories):
            start_state = create_initial_state(question)
            strategy = traj["fn"]()
            final_state = run_strategy(engine, start_state, strategy)
            last_state = final_state

            final_answer = final_state.get("answer") or ""
            judged_correct, _ = judge.judge(final_answer, ground_truth, question)
            measured_joules = float(final_state.get("total_joules", 0.0))

            attempt_records.append(
                {
                    "trajectory_id": traj_idx,
                    "trajectory_name": traj["name"],
                    "estimated_cost": float(strategy_cost(strategy, cost_table)),
                    "measured_joules": measured_joules,
                    "is_correct": bool(judged_correct),
                    "status": final_state.get("status", ""),
                }
            )

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
                    "optimal_trajectory_id": chosen_id,
                    "joules_spent": joules_spent,
                    "is_correct": is_correct,
                    "execution_mode": args.execution_mode,
                    "attempts": attempt_records,
                    "history": best_correct_state.get("history", []) if is_correct and best_correct_state is not None else [],
                }
            ],
        )

        processed_count += 1
        if processed_count % 10 == 0:
            logging.info("Processed %s/%s", processed_count, args.limit)

    logging.info("Done. Output saved to %s", args.output)


if __name__ == "__main__":
    main()
