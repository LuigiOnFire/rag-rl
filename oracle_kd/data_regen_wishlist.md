Here is the updated and expanded **Data Regeneration Wish List**. It integrates the mandate for real-world document contexts within the strided simulation loops, isolates the software-efficiency gap for keyword searches, and introduces explicit metrics for tracking environment state growth.

---

### 1. The Strided Batching Loop (Cache Isolation & Real-Corpus Grounding)

* **What to change:** Invert the execution loops to process queries in small interleaved blocks (e.g., blocks of 20 queries) rather than evaluating all 8 trajectories back-to-back on a single query. Crucially, discard any placeholder text structures or short toy strings used during context initialization or trajectory pre-estimation. The data framework must ingest raw, multi-paragraph document chunks directly from your active evaluation corpora (HotpotQA and the 21M-chunk DPR index).
* **Why it matters:** Interleaving blocks naturally overflows the 8,192-token GPU context window, completely evicting the KV-cache of previous queries before a new trajectory loops back to it. Forcing the loop to process actual, full-scale document text ensures that host memory allocations, tensor sequence lengths, and KV-cache hydration profiles reflect real data center workloads. This prevents underestimating execution costs during the initial scheduling pass.

### 2. Full Fidelity Logging Schema Upgrade

* **What to change:** Modify the JSON serialization block to nest the full `GreenHistoryItem` step-by-step logs inside *every individual attempt* within the `attempts` array, rather than only saving the step history for the overall winning trajectory.
* **Why it matters:** This unlocks a much larger data pool for Section 3.2 (Micro-Action Hardware Breakdown). You will be able to capture the raw energy cost of SLM/LLM generations and searches across *every* executed path, maximizing your sample size ($N$).

### 3. Native Wall-Clock Latency Telemetry

* **What to change:** In your execution context manager or worker loops, wrap actions in a standard Python timer (`time.perf_counter()`) and log a `duration_seconds` or `measured_latency` key inside each trajectory attempt object.
* **Why it matters:** This provides the missing telemetry required to compute the Pearson correlation coefficient between Wall-Clock Latency and Physical Joules for Section 3.3, proving whether or not energy profiling provides distinct signal over a simple timer.

### 4. Isolated Hardware Baseline Control Group

* **What to change:** Programmatically trigger 50 standalone iterations of a completely cold vector search (`RET_VEC`) and a cold generation pass (`GEN_LLM`) at the very beginning of the master execution script, dumping them into a separate `hardware_baseline.json` control file.
* **Why it matters:** This gives you a pristine, unpolluted baseline of the raw physical tax of individual operations to benchmark your active, multi-hop RAG trajectories against. This is distinct from Item 1: Item 1 ensures live trajectories use genuine data to avoid profiling distortion, while Item 4 isolates the underlying hardware components completely outside of an agent loop.

### 5. Embedded Dataset Metadata String

* **What to change:** Programmatically inject a strict, normalized metadata field (e.g., `"source": "squad"` or `"source": "nq"`) directly into the JSON dictionary payload at generation time.
* **Why it matters:** Eliminates the dependency on file-naming string conventions during post-processing analysis, making your downstream analytics scripts completely robust against directory structures or filename changes.

### 6. Accelerated Sparse Retrieval Profiling Baseline (The Keyword Search Fix)

* **What to change:** Integrate a parallel profiling branch that logs an optimized, compiled keyword search alternative (or a lean pre-compiled execution script) alongside the native `rank_bm25` run. Tag these entries distinctly in your logs (e.g., `RET_KEY_PURE` vs. `RET_KEY_ACCEL`).
* **Why it matters:** This allows you to explicitly isolate the exact Joule and latency tax caused by pure-Python single-threaded execution bottlenecks versus optimized sparse search configurations. It provides the empirical proof needed to demonstrate how unaccelerated software dependencies can accidentally trigger massive data center energy anomalies.

### 7. Step-Level Environment State Size Tracking

* **What to change:** Inject a tracking block into the telemetry loop that records the precise dimension of the state space at step $t$ ($S_t$). Log explicit integer values for `input_token_count`, `accumulated_context_tokens`, and the raw byte size of all appended document text blocks.
* **Why it matters:** This data is required to substantiate your Analytical Cost Model ($E_{\text{total}} = \sum E(A_t, S_t)$). By recording token growth at every transition, you can map exactly how sequence-length extensions drive quadratic variations in generation energy and VRAM consumption across lengthy agentic loops.