# CostAware-RAG

**Goal:** Create a local, energy-efficient RAG (Retrieval-Augmented Generation) agent that learns to minimize Joules per correct answer.

This project implements a reinforcement learning environment where an agent composed of Small Language Models (SLMs) and Large Language Models (LLMs) learns to navigate a search space of actions (Generate, Retrieve, Grade, Rewrite, Decompose) to answer questions while minimizing energy consumption.

## 🏗️ Hardware Stack

The architecture is designed to balance performance and energy efficiency by offloading simple tasks to smaller models.

*   **Manager**: `Llama-3.2-1B-Instruct`
    *   **Role**: Decision Maker, State Holder. Decides the next action based on the current state.
*   **Worker S (Small)**: `Llama-3.2-1B-Instruct` (via Ollama)
    *   **Role**: Rewriting queries, Grading relevance, Summarizing context.
*   **Worker L (Large)**: `llama3:8b` (via Ollama)
    *   **Role**: Complex Reasoning, Deep Decomposition, Final Answer Generation.
*   **Retriever**:
    *   **Dense**: `BAAI/bge-base-en-v1.5`
    *   **Sparse**: `BM25Okapi`

## ⚡ The Atomic Action Space

The agent operates using 10 discrete actions, each with an associated energy cost (Joules).

| ID | Mnemonic | Description | Model Used |
| :--- | :--- | :--- | :--- |
| **0** | `GEN_SLM` | Generate Answer (Cheap) | 1B |
| **1** | `GEN_LLM` | Generate Answer (Expensive) | 8B |
| **2** | `RET_KEY` | Keyword Search (BM25) | CPU |
| **3** | `RET_VEC` | Vector Search (Dense) | CPU/GPU |
| **4** | `GRD_SLM` | Check Relevance | 1B |
| **5** | `GRD_LLM` | Hallucination Check | 8B |
| **6** | `RWT_SLM` | Query Rewrite | 1B |
| **7** | `DEC_SLM` | Simple Decomposition | 1B |
| **8** | `DEC_LLM` | Deep Decomposition | 8B |
| **9** | `FAIL` | Abort/Give Up | N/A |

## 🚀 Setup & Installation

### Prerequisites

1.  **Python 3.10+**
2.  **Ollama**: Used to serve the local LLMs.
    *   Install Ollama: [https://ollama.com](https://ollama.com)
    *   Pull the required models:
        ```bash
        ollama pull llama3:8b
        ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest
        ```

### Installation

Clone the repository and install the Python dependencies:

```bash
pip install -r requirements.txt
```

## 🛠️ Usage

### 1. Calibration (Measure Energy Costs)
Before generating data, the system needs to measure the real-world energy consumption (in Joules) for each primitive action on your specific hardware.

Run the calibration script:

```bash
python scripts/01_calibrate.py
```

*   **What it does:** Runs each action multiple times, measures energy using `codecarbon`, and calculates the average Joules/Action.
*   **Output:** `data/meta/cost_table.json`

### 2. Generate Training Data (The Oracle)
The Oracle generates "Golden Trajectories" (sequences of actions) that solve questions with the minimum possible energy cost. It uses a **Constrained Best-First Search** and **Ephemeral Retrieval**.

Run the generation script:

```bash
python scripts/02_generate.py
```

*   **What it does:** 
    1.  Streams samples from the dataset (e.g., HotpotQA).
    2.  Builds an **Ephemeral Index** (instant in-memory vector DB) for each sample.
    3.  Finds the cheapest valid path to the correct answer using cost-aware search.
    4.  Validates answers using a **Soft Judge** (String Limit -> F1 -> LLM).
*   **Output:** `data/trajectories/gold_trajectories.jsonl`

## 📂 Project Structure

The project separates logic (`src`) from execution (`scripts`) and artifacts (`data`).

```
cost-aware-rag/
├── data/                       # ARTIFACTS
│   ├── meta/cost_table.json    # Energy Prices
│   └── trajectories/           # Training Data (Output)
├── src/                        # LIBRARY
│   ├── agent/                  # INFERENCE LOGIC
│   │   ├── actions.py          # Atomic Action Definitions (0-9)
│   │   └── workers.py          # Wrappers for Llama-1B, Llama-8B (Argument Generators)
│   ├── env/                    # SIMULATION
│   │   ├── retriever.py        # The "Ephemeral" Retriever (In-Memory per sample)
│   │   └── state.py            # GreenState Schema
│   ├── oracle/                 # DATA GEN LOGIC
│   │   ├── search.py           # Best-First Search & Grammar Constraints
│   │   ├── judge.py            # Soft Validation (Tiered: String -> F1 -> LLM)
│   │   └── calibrator.py       # Energy Measurement Class
│   └── data/                   # DATALOADERS
│       └── hotpot.py           # HotpotQA Streamer
├── scripts/                    # EXECUTABLES
│   ├── 01_calibrate.py         # Runs Calibrator
│   └── 02_generate.py          # Runs Oracle Generation
├── requirements.txt
└── README.md
```

## 🛡️ License

MIT
