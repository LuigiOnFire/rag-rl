# Beyond Tokens

This repository contains the reproducibility code for **Beyond Tokens:
Benchmarking the Energy Cost of Retrieval-Augmented Generation Pipelines**.

The benchmark studies RAG pipelines as sequences of discrete, measurable
actions. It evaluates the trade-off between answer quality, measured energy in
Joules, and latency across open-domain question-answering datasets. GPU power
is measured through driver interfaces, while host-side infrastructure is
approximated through system-wide compute footprints.

The included energy-data pipeline generates route measurements and labels each
question with the least expensive successful action sequence. These labels
support analysis of sparse and dense retrieval, language-model generation,
reasoning, decomposition, and unnecessary escalation of simple queries.

## Workflow

The numbered stages are the reproducibility order:

```text
00_build_corpus -> 01_build_sparse_index -> 02_build_dense_index
    -> 03_calibrate -> 04_generate_energy_data
```

Merge, parsing, and analytics scripts are optional post-processing tools and
intentionally use descriptive names.

## Requirements

- Python 3.10 or newer
- CUDA-capable hardware for dense indexing and model inference
- Ollama for local language-model inference
- Java and Pyserini for the Lucene sparse index
- The dependencies in `requirements.txt`

The benchmark uses two Ollama model roles:

- `LLM_MODEL`: an 8B-class model for expensive generation and reasoning
- `SLM_MODEL`: a 1B-class model for lower-cost actions
- `BAAI/bge-base-en-v1.5`: the dense retrieval embedding model

### Python environment

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install Java if it is not already available. Pyserini requires a working Java
runtime for the sparse index.

### Environment variables

Create a local `.env` file in the repository root. This file is intentionally
not tracked and should not contain committed credentials:

```dotenv
LLM_MODEL=llama3:8b
SLM_MODEL=hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest
```

Load it into the current shell before running Python scripts:

```bash
set -a
source .env
set +a
```

If your model names differ, change the two values in `.env`. The defaults in
`src/agent/workers.py` match the example above.

### Ollama

Install Ollama using the instructions at <https://ollama.com>, start the local
server, and pull both models named in `.env`:

```bash
ollama serve
```

In a second terminal, from the repository root:

```bash
set -a
source .env
set +a
ollama pull "$LLM_MODEL"
ollama pull "$SLM_MODEL"
```

Keep `ollama serve` running while calibrating or generating energy data.

## Running the Pipeline Locally

The scripts use paths relative to the repository root.

### 00. Build the corpus

```bash
python scripts/00_build_corpus.py --corpus-type fullwiki
```

For SQuAD/DPR retrieval, use `--corpus-type squad_wiki` where supported.

### 01. Build the sparse index

```bash
python scripts/01_build_sparse_index.py --corpus-type fullwiki
```

This creates `data/indices/fullwiki_index/`.

### 02. Build the dense index

```bash
python scripts/02_build_dense_index.py --corpus-type fullwiki
```

This creates the FAISS index under `data/meta/`.

### 03. Calibrate action costs

```bash
python scripts/03_calibrate.py
```

Calibration writes `data/meta/cost_table.json`. Because energy depends on
hardware, regenerate the table on the target system when exact measurements
are required.

### 04. Generate energy data

Run a one-example smoke test first:

```bash
python scripts/04_generate_energy_data.py --dataset-name hotpotqa --limit 1
```

For a full run, increase `--limit` and select `hotpotqa`, `squad`, or `nq`.
The generator writes timestamped training CSV, trajectory JSONL, and per-query
trace files under `data/energy_data/`.

For HPC users, equivalent Slurm launchers are available in `hpc/`:

```text
hpc/00_build_corpus.sbatch
hpc/01_build_sparse_index.sbatch
hpc/02_build_dense_index.sbatch
hpc/03_calibrate.sbatch
hpc/04_generate_energy_data.sbatch
```

They are optional wrappers around the same numbered scripts and require a
site-specific Apptainer, Slurm, Java, and environment configuration.

## Checkpoint Data

The published checkpoint datasets are stored in:

```text
data/energy_data/training/
data/energy_data/trajectories/
```

They contain merged query labels and trajectory histories used by the paper's
analyses. The merge utilities are:

```text
scripts/merge_query_csv.py
scripts/merge_traj_json.py
scripts/merge_energy_data.py
```

Analytics live in `analytics/`; `scripts/energy_data_gen_parse.py` provides summary
parsing for generated trajectory records.

## External Artifacts

The full-wiki corpus and retrieval indexes are intentionally not stored in Git.
They can require tens of gigabytes, especially the DPR FAISS index. The
preparation scripts regenerate the required assets under:

```text
data/meta/
data/indices/
```

Generated logs, model weights, Ollama model data, Slurm outputs, and temporary
run artifacts are also excluded. Do not commit credentials, `.env` files, or
local model caches.

## Repository Layout

```text
scripts/       Numbered pipeline and descriptive post-processing tools
src/           Agent actions, datasets, environment, retrieval, and judging
analytics/     Aggregate trajectory and action-level analysis
hpc/           Slurm launchers matching the numbered pipeline
data/energy_data/   Published energy data and generated route outputs
data/meta/     Locally generated corpora, indexes, and calibration metadata
docs/          Supporting methodological notes
```

## License

MIT