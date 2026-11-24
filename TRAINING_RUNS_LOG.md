### GRPO Training Run Log & Analysis

**Phase 1: The Baseline & The Iron Anchor (High Learning Rate)**

```bash
sbatch hpc/run_05_train_grpo.sbatch --learning_rate 2.0e-6 --kl_coef 0.04 --danger_pay_base 0.10 --run_name "greenrag-grpo-baseline" --output_dir "models/green-rag-grpo-baseline"

```

* **Hypothesis:** A standard GRPO configuration with an empirically calibrated learning rate and a standard generative KL tether ($\beta = 0.04$) will allow the agent to optimize for the Danger Pay reward safely.
* **Finding:** Catastrophic KL Explosion. The KL divergence violently spiked to >3.5 within the first 100 steps, indicating the policy gradient updates were destroying the SFT prior and tearing the underlying language syntax apart.

```bash
sbatch hpc/run_05_train_grpo.sbatch --learning_rate 2.0e-6 --kl_coef 0.08 --danger_pay_base 0.10 --run_name "greenrag-grpo-kl08" --output_dir "models/green-rag-grpo-kl08"

```

* **Hypothesis:** Doubling the KL coefficient to a strict 0.08 (The "Iron Anchor") will force the model to stay close to the SFT baseline and prevent the weights from melting.
* **Finding:** The anchor failed. The KL still exploded to 3.15 at Step 90. This proved that a learning rate of `2.0e-6` was structurally too massive for the variance of the environment at a batch size of 2 (effective 8). The gradients acted like a sledgehammer, overpowering even heavy regularization.

**Phase 2: The Over-Correction & Paralysis (Low Learning Rate)**

```bash
sbatch hpc/run_05_train_grpo.sbatch --learning_rate 1.0e-6 --kl_coef 0.04 --danger_pay_base 0.05 --run_name "greenrag-grpo-lr1e6-dp05" --output_dir "models/green-rag-grpo-lr1e6-dp05"

sbatch hpc/run_05_train_grpo.sbatch --learning_rate 5.0e-7 --gradient_accum 4 --kl_coef 0.04 --danger_pay_base 0.05 --run_name "greenrag-grpo-micro-lr5e7" --output_dir "models/green-rag-grpo-micro-lr5e7"

sbatch hpc/run_05_train_grpo.sbatch --learning_rate 1.0e-6 --gradient_accum 8 --kl_coef 0.04 --danger_pay_base 0.05 --run_name "greenrag-grpo-smooth-acc8" --output_dir "models/green-rag-grpo-smooth-acc8"

```

* **Hypothesis:** Shrinking the learning rate (and optionally increasing gradient accumulation to 8) will stop the sledgehammer effect, allowing safe, gradual optimization.
* **Finding:** Severe Policy Paralysis. The KL divergence flatlined between 0.001 and 0.01. Because the KL penalty pushes back against change, the microscopic learning rates lacked the momentum to shift the policy at all. The agents were mathematically frozen.
* **The Paradigm Shift:** We realized that a high KL divergence (e.g., 2.0) is actually mathematically expected and necessary for a discrete 1-token router shifting probabilities from one categorical action to another. Applying generative sequence RLHF limits (KL < 0.1) was artificially stifling the model.

**Phase 3: The Unleashed Runs & The Loophole Discovery (High LR, Low/Zero KL)**

```bash
sbatch hpc/run_05_train_grpo.sbatch --learning_rate 5.0e-6 --gradient_accum 4 --kl_coef 0.005 --danger_pay_base 0.05 --run_name "greenrag-grpo-unleashed-kl005" --output_dir "models/greenrag-grpo-unleashed-kl005"

```

* **Hypothesis:** By raising the learning rate back to `5.0e-6` but dropping the KL penalty to a near-zero `0.005` tether, the agent is unchained to freely alter its routing probabilities relying solely on PPO clipping (`CLIP_EPS = 0.2`) for safety.
* **Finding:** The routing metrics looked perfect (KL rose to a healthy 1.0–1.6, decomposition settled at a Pareto-efficient 20%), but the greedy evaluation accuracy plummeted from 0.45 to 0.15.
* **Root Cause:** Reward Hacking. The agent discovered an "infinite money glitch" in the reward function. Because the script clamped energy penalties at zero (`max(0.0, 0.0 - joule_penalty)`), the agent realized it could intentionally fail via the expensive Decomposition pipeline to farm a guaranteed +0.05 Danger Pay subsidy without paying any energy costs.

**Phase 4: The V2 Runs (The Loophole Fix)**
*   **Hypothesis:** The V1 "Unleashed" runs achieved good telemetry but collapsed in evaluation because the energy penalty was floored at `0.0`. The agent was farming Danger Pay without consequence. Removing the floor will force true efficiency.
*   **Finding:** The environment was fixed, but it revealed a new behavioral collapse...

sbatch hpc/run_05_train_grpo.sbatch --learning_rate 5.0e-6 --gradient_accum 4 --kl_coef 0.0 --danger_pay_base 0.05 --run_name "greenrag-grpo-v2-kl0" --output_dir "models/greenrag-grpo-v2-kl0"

sbatch hpc/run_05_train_grpo.sbatch --learning_rate 5.0e-6 --gradient_accum 4 --kl_coef 0.005 --danger_pay_base 0.05 --run_name "greenrag-grpo-v2-kl005" --output_dir "models/greenrag-grpo-v2-kl005"

sbatch hpc/run_05_train_grpo.sbatch --learning_rate 5.0e-6 --gradient_accum 4 --kl_coef 0.0 --danger_pay_base 0.10 --run_name "greenrag-grpo-v2-kl0-dp10" --output_dir "models/greenrag-grpo-v2-kl0-dp10"

**Phase 5: The Cowardice Collapse (V3 Tuning)**
*   **Date:** October 8, 2026
*   **Finding:** Closing the zero-floor reward loophole caused the agent to become terrified of energy costs. `tools/generate_pct` hit 100% across all runs, and `mean_joules` dropped to 700J. 
*   **Root Cause:** The Math of Cowardice. At `joule_lambda=2.0e-5`, failing via `Generate` (-0.02) was mathematically less punishing than failing via `Decompose` (-0.03). The agent realized that immediately guessing and failing was the safest way to avoid massive, uncapped looping penalties.

**The Fix:** Lowered `JOULE_LAMBDA` in `05_train_grpo.py` from `5.0e-5` to `2.0e-5` to make energy taxes 60% cheaper, leveling the risk/reward ratio. 

**V3 Overnight Commands:**
sbatch hpc/run_05_train_grpo.sbatch --learning_rate 5.0e-6 --gradient_accum 4 --kl_coef 0.0 --danger_pay_base 0.05 --joule_lambda 2.0e-5 --run_name "greenrag-grpo-v3-pure" --output_dir "models/greenrag-grpo-v3-pure"

sbatch hpc/run_05_train_grpo.sbatch --learning_rate 5.0e-6 --gradient_accum 4 --kl_coef 0.005 --danger_pay_base 0.05 --joule_lambda 2.0e-5 --run_name "greenrag-grpo-v3-tether" --output_dir "models/greenrag-grpo-v3-tether"

sbatch hpc/run_05_train_grpo.sbatch --learning_rate 5.0e-6 --gradient_accum 4 --kl_coef 0.005 --danger_pay_base 0.10 --joule_lambda 2.0e-5 --run_name "greenrag-grpo-v3-high-subsidy" --output_dir "models/greenrag-grpo-v3-high-subsidy"

**Phase 6: Risk-Bounded Exploration (V4 Tuning)**
*   **The Fix:** Lowered `JOULE_LAMBDA` to `1.0e-5` and added a `max(-0.05, reward)` safety net to the failure calculation. This creates a perfect risk hierarchy:
    *   Failed Decompose (~4000J): `+0.01`
    *   Failed Generate (~1000J): `-0.01`
    *   Looping (>5000J): `-0.05` (Bounded)
*   **V4 Commands:**
sbatch hpc/run_05_train_grpo.sbatch --learning_rate 5.0e-6 --gradient_accum 4 --kl_coef 0.005 --danger_pay_base 0.05 --joule_lambda 1.0e-5 --run_name "greenrag-grpo-v4-tether" --output_dir "models/greenrag-grpo-v4-tether" # FAILED, collapsed to only generate quickly, replaced by pure smooth

sbatch hpc/run_05_train_grpo.sbatch --learning_rate 5.0e-6 --gradient_accum 4 --kl_coef 0.0 --danger_pay_base 0.05 --joule_lambda 1.0e-5 --run_name "greenrag-grpo-v4-pure" --output_dir "models/greenrag-grpo-v4-pure" # failed, was genearting only

sbatch hpc/run_05_train_grpo.sbatch --learning_rate 5.0e-6 --gradient_accum 4 --kl_coef 0.005 --danger_pay_base 0.10 --joule_lambda 1.0e-5 --run_name "greenrag-grpo-v4-high-subsidy" --output_dir "models/greenrag-grpo-v4-high-subsidy" # we decided decomposition was polluting the reward space too much

sbatch hpc/run_05_train_grpo.sbatch --learning_rate 2.0e-6 --gradient_accum 8 --kl_coef 0.0 --danger_pay_base 0.05 --joule_lambda 1.0e-5 --run_name "greenrag-grpo-v4-pure-smooth" --output_dir "models/greenrag-grpo-v4-pure-smooth" # also failed, generated only

**Phase 7 (The Clean Baseline)**

* In Phases 1–6, hyperparameters were compensating for broken mathematical implementations:
    * The high LR (5.0e-6) was required because step_loss was being diluted by 1 / len(traj).
    * The KL exploded because log-softmax was computed across the 151,643-token vocabulary rather than the 9-action space.
    * The cowardice collapse happened because failures had unequal rewards.
    * Danger pay removed. We wish to avoid corrupting the reward space.

* With those bugs resolved, the gradient updates should be cleaner and more potent.

sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 2.0e-6 \
  --kl_coef 0.01 \
  --joule_lambda 2.0e-5 \
  --gradient_accum 4 \
  --run_name "greenrag-grpo-v5-anchor" \
  --output_dir "models/greenrag-grpo-v5-anchor"

  sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 2.0e-6 \
  --kl_coef 0.01 \
  --joule_lambda 1.0e-5 \
  --gradient_accum 4 \
  --run_name "greenrag-grpo-v5-gentle-tax" \
  --output_dir "models/greenrag-grpo-v5-gentle-tax"

  sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 2.0e-6 \
  --kl_coef 0.01 \
  --joule_lambda 2.0e-5 \
  --gradient_accum 8 \
  --run_name "greenrag-grpo-v5-smooth-acc8" \
  --output_dir "models/greenrag-grpo-v5-smooth-acc8"

### Phase 7: The "KL Tax" Deadlock

**Date:** October 8, 2026
**Finding:** The V5 runs failed to explore Decomposition, despite the environment math being fixed. Telemetry revealed the agent was executing `Keyword -> Generate` (Standard Single-Hop RAG) for almost every query.
**Root Cause:** The KL Tax. Because the model's SFT prior heavily favored Single-Hop RAG, executing `Keyword -> Generate` incurred zero KL penalty. Exploring the `Decompose` path generated a KL divergence of ~4.5. Even with `kl_coef` reduced to `0.01`, the resulting KL Tax (-0.045) entirely wiped out the Danger Pay subsidy (+0.01 net), making exploration mathematically worse (-0.035) than safely failing via Single-Hop RAG (-0.015).

### Phase 8: The Purified Ledger & Hardware Unblocking

**Date:** October 11, 2026

**Context & Previous Failures:**
Previous attempts were bottlenecked by a combination of hardware constraints and mathematical paradoxes within the environment.

1. **The Hardware Deadlock:** Runs like `v4-high-subsidy` stalled to 3-minute iteration times because `OLLAMA_GPU_OVERHEAD` was set too high (14GB), forcing the final 2 layers of the Qwen-32B model to spill over into CPU RAM.
2. **The "Cowardice" Collapse:** Penalizing energy on failed trajectories taught the agent to fail as quickly and cheaply as possible (100% `Generate` actions) to avoid massive negative variance.
3. **The "Reward Hacking" Paradox:** Introducing "Danger Pay" corrupted the Markov Decision Process by applying a static subsidy, teaching the agent to intentionally fail complex trajectories just to harvest the base reward.

**The Fixes:**
The environment is now a pristine mathematical ledger.

* **Hardware:** `OLLAMA_GPU_OVERHEAD` is reduced to 12GB (`12884901888`), ensuring all 65/65 layers remain in VRAM for rapid inference.
* **Reward:** Danger Pay is dead. **Flat Failure Penalties** are active. All failures receive a static `-0.10` reward. Energy (Joules) is only taxed on *successful* trajectories to find the Pareto efficiency. This permanently removes the agent's fear of exploring complex routes, while simultaneously removing any financial incentive to farm failures.
* **Mathematical Stability:** The KL divergence is calculated exactly over the discrete 9-action space (eliminating 150k-token vocab variance), and trajectory length dilution (`len(traj)`) has been removed from the gradient updates.

**V8 HPC Commands (The 3-Slot Strategy):**
With the math purified, we utilize the three HPC slots to test the optimal balance of exploration and gradient stability.

**Slot 1: The Clean Baseline (Pure PPO)**
With Flat Failure penalties protecting the agent from massive negative variance, we can safely drop the KL coefficient to 0.0. This completely severs the "KL Tax" that tied the agent to its single-hop SFT prior, allowing it to freely explore the decomposition space purely based on energy efficiency and success rates.

```bash
sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 2.0e-6 \
  --kl_coef 0.0 \
  --joule_lambda 1.0e-5 \
  --gradient_accum 4 \
  --run_name "greenrag-grpo-v8-pure" \
  --output_dir "models/greenrag-grpo-v8-pure"

```

**Slot 2: The Micro-Tether**
This acts as a gentle safety net. A microscopic `0.001` KL penalty provides just enough gradient pressure to prevent the policy from wandering into completely nonsensical random token generation, but the "KL Tax" is so small that it will not artificially suppress the +0.01 advantage of a successful decomposition.

```bash
sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 2.0e-6 \
  --kl_coef 0.001 \
  --joule_lambda 1.0e-5 \
  --gradient_accum 4 \
  --run_name "greenrag-grpo-v8-micro-tether" \
  --output_dir "models/greenrag-grpo-v8-micro-tether"

```

**Slot 3: The Slow & Steady Variance Reducer**
Zero KL tether, but doubling the gradient accumulation to 8. The agent calculates advantages across 16 questions (64 rollouts) before executing a single gradient update. This ensures that any remaining environment noise (e.g., the downstream 32B model hallucinating on a perfectly routed query) is completely smoothed out over the larger batch size, guaranteeing highly stable policy shifts.

```bash
sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 2.0e-6 \
  --kl_coef 0.0 \
  --joule_lambda 1.0e-5 \
  --gradient_accum 8 \
  --run_name "greenrag-grpo-v8-pure-smooth" \
  --output_dir "models/greenrag-grpo-v8-pure-smooth"

```

sbatch hpc/run_05_train_grpo.sbatch --learning_rate 2.0e-6 --kl_coef 0.04 --danger_pay_base 0.10 --run_name "greenrag-grpo-baseline" --output_dir "models/green-rag-grpo-baseline"



### Phase 6: Trajectory Forcing
Next we implement trajectory forcing and lower the joule lambda and the learning rate. I'll put kl coeff a bit higher, closer to where it was in earlier versions. With this setup I will try a range of learning rates.

sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 1.0e-7 \
  --kl_coef 0.0 \
  --joule_lambda 1.0e-7 \
  --gradient_accum 8 \
  --run_name "greenrag-grpo-v6-low-lr" \
  --output_dir "models/greenrag-grpo-v6-low-lr"

sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 5.0e-7 \
  --kl_coef 0.0 \
  --joule_lambda 1.0e-7 \
  --gradient_accum 8 \
  --run_name "greenrag-grpo-v6-med-lr" \
  --output_dir "models/greenrag-grpo-v6-med-lr"

sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 5.0e-6 \
  --kl_coef 0.0 \
  --joule_lambda 1.0e-7 \
  --gradient_accum 8 \
  --run_name "greenrag-grpo-v6-high-lr" \
  --output_dir "models/greenrag-grpo-v6-high-lr"

### Phase 7: Termination Urgency, Length Normalization & KL Anchoring
**Phase 6 Findings (`v6-high-lr` Analysis):**

* **Proof of Concept:** Trajectory forcing (`Gen 1/4` forced to `ACTION_DEC_LLM` at Step 0) successfully broke Advantage Starvation, driving evaluation accuracy up to **0.65** before collapsing back to **0.35–0.40**.

* **Root Cause of Collapse (`[6 -> 3 -> ... -> 3]` Loop):** By Step 647, `Gen 1/4` locked into an infinite dense retrieval loop (`[6 -> 3 -> 3 -> 3 -> 3 -> 3 -> 3 -> 3 -> 3 -> 3 -> 3 -> 3 -> 3 -> 3 -> 3]`) on 100% of questions, failing to call `Generate` before hitting the 15-step limit.

* **Why It Locked In:**
1. **Unnormalized Trajectory Gradients:** `step_loss` in `grpo_loss` was not divided by `len(traj)`, causing repeated intermediate retrieval steps (`Action 3`) in successful multi-step chains to receive $3\times$–$10\times$ larger cumulative gradient updates than the single terminal `Generate` step (`Action 1`).
2. **Softmax Saturation (`kl_coef = 0.0`):** With no KL tether, `KL` drifted to `1.0–2.96` nats and $\pi_\theta(3 \mid s_{\text{decomp}})$ saturated to $\approx 1.0$, causing the policy gradient $\propto (1 - \pi_\theta(a))$ on Steps 1–14 to vanish to zero.

3. **Markov State Stagnation:** Repeated retrievals returned identical documents, and `format_state_for_prompt` lacked a step counter, making $s_{t+1}$ indistinguishable from $s_t$.
4. **No Termination Urgency:** Timeouts with no answer (`PRED=''`) received the exact same `-0.100` penalty as an honest wrong answer, creating no incentive to commit to an answer before Step 15.

**Code & Hyperparameter Changes:**

1. **Explicit Step Horizon in Prompt (`src/agent/prompts.py`):** Updated `format_state_for_prompt` to inject `Step: {current_step}/{max_steps} (Steps Remaining: {steps_remaining})` in the header and immediately above the `Action:` trigger, plus step indices (`[Step X]`) on each history entry to break Markov state symmetry across repeated tool calls.
2. **Asymmetric Timeout Penalty (`05_train_grpo.py`):** Updated `compute_reward` to penalize non-termination/empty answers (`-0.25`) harder than incorrect answers (`-0.10`). Added two safeguards in `grpo_loss`:
* Clamped advantages to `max=0.0` when all rollouts in a group fail (`reward_tensor.max() <= 0.0`), preventing failed `[2 -> 1]` guesses (`-0.10`) from receiving positive advantages against a `-0.25` timeout.
* Exempted `step_i == 0` from timeout penalties (`raw_r <= -0.20`) so downstream looping (`-> 3 -> 3 -> 3`) is punished without suppressing the root `Decompose` decision.


3. **Restored KL Anchor (`--kl_coef 0.01`):** Raised `kl_coef` from `0.0` back to `0.01` to prevent logit runaway/softmax saturation and retain the SFT reference policy's ability to transition from retrieval to generation.
4. **Trajectory-Length Loss Normalization (`05_train_grpo.py`):** Divided `step_loss` by `len(traj)` during backpropagation (`step_loss = (pg_loss + kl_loss) * (1.0 / (len(traj) * num_valid * batch_scale))`) so long trajectories do not disproportionately amplify intermediate retrieval actions over terminal generation actions.

```bash
# Relaunch High LR with KL anchor, timeout penalty, step horizon, and length-normalized loss
sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01 \
  --joule_lambda 1.0e-7 \
  --gradient_accum 8 \
  --run_name "greenrag-grpo-v7-high-lr" \
  --output_dir "models/greenrag-grpo-v7-high-lr"

# Optional Mid-High LR (2.0e-6) if 5.0e-6 still shows volatility
sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 2.0e-6 \
  --kl_coef 0.01 \
  --joule_lambda 1.0e-7 \
  --gradient_accum 8 \
  --run_name "greenrag-grpo-v7-2e6-lr" \
  --output_dir "models/greenrag-grpo-v7-2e6-lr"

```

# Thrifty RAG: GRPO Training Journal & Experiment Log

## 1. Architectural & Mathematical Debugging Log
- **Advantage Starvation & The EV Trap**: Initial multi-step decomposition success rates (~15%) lagged far behind single-hop retrieval/generation (~40%), causing the optimizer to prematurely prune complex tools. Resolved via **Stratified Exploration / Trajectory Forcing** (Forcing Rollout 0 to execute `ACTION_DEC_LLM` at Step 0 while leaving rollouts 1–3 organic).
- **Environment & Software Fixes**:
  - Solved `scontrol` missing binary spam by scrubbing all `SLURM_*` environment variables right before launching Python/`wandb`.
  - Resolved VRAM deadlocks and Ollama/PyTorch thrashing by adding frequent `torch.cuda.empty_cache()` calls per question iteration.
- **GRPO Advantage & Loss Normalization**:
  - Discovered that standard z-score normalization (`(R - mean) / std`) divides by microscopic $\sigma_R$ on all-pass batches (e.g., `0.999` vs `1.000`), creating an unintended `-1.50` penalty on successful decomposition rollouts. Transitioned to **Dr. GRPO mean-centered advantages (`R - \bar{R}`)**.
  - Restored `kl_coef = 0.01` to anchor policies to the SFT reference prior and prevent softmax saturation.
  - Added trajectory length normalization (`/ len(traj)`) to `step_loss` to prevent intermediate multi-hop retrieval steps from accumulating $3\times$–$10\times$ larger cumulative gradients than terminal generation steps.

## 2. Prompt Engineering & Evaluation Improvements
- **Step Horizon Tracker**: Injected `[Step X/15]` directly above the action trigger window and indexed history items (`[Step 1]`, `[Step 2]`) to break Markov state repetition during infinite retrieval loops.
- **Deterministic Validation Set**: Pre-loaded a fixed set of validation samples (`fixed_eval_samples`) once at initialization to eliminate evaluation noise across checkpoints.

---

## 3. Experiment History & Configurations

### Phase 6: Ablation & Scaling Runs (`v6`)
- **Setup**: Launched three ablation runs across learning rates (`low-lr` at `1.0e-7`, `med-lr` at `5.0e-7`, and `high-lr` at `5.0e-6`).
- **Observations**: 
  - `high-lr` achieved a peak evaluation accuracy of **0.65**, proving multi-step decomposition works without Danger Pay, but eventually crashed into a dense retrieval loop (`[6 -> 3 -> 3 -> ... -> 3]`) due to logit saturation and unnormalized trajectory loss.
  - All three learning rates followed identical KL divergence scaling curves proportional to their learning rates.

### Phase 7: Production Runs (`v7`)
- **Key Modifications**:
  1. Explicit step horizon injected into prompts (`src/agent/prompts.py`).
  2. Asymmetric timeout penalty (`-0.25` for non-termination vs `-0.10` for incorrect guesses) with step-0 root action protection.
  3. Restored KL anchor (`--kl_coef 0.01`).
  4. Trajectory-length loss normalization (`1.0 / (len(traj) * num_valid * batch_scale)`).
  5. Dr. GRPO mean-centered advantages.
---
---
- **Launch Commands**:
```bash
# High-LR Production Run
sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01 \
  --joule_lambda 1.0e-7 \
  --gradient_accum 8 \
  --run_name "greenrag-grpo-v7-5e-6-lr" \
  --output_dir "models/greenrag-grpo-v7-5e-6-lr"

# Mid-LR Production Run
sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 2.0e-6 \
  --kl_coef 0.01 \
  --joule_lambda 1.0e-7 \
  --gradient_accum 8 \
  --run_name "greenrag-grpo-v7-2e6-lr" \
  --output_dir "models/greenrag-grpo-v7-2e-6-lr"

* **Phase 7 Key Findings**:
1. **Zero Collapse**: Both runs completed 1,000 steps with **0.0% evaluation timeouts**. The `-0.25` timeout penalty + step horizon prompt + `kl_coef=0.01` permanently solved the `v6` infinite-loop collapse.

2. **Emergent Discovery of `[2 -> 1]` (`RET_KEY -> GEN_LLM`)**: Between Step 500 and Step 750, `v7-5e-6-lr` escaped the human-designed SFT prior (`[3 -> 1]`) and discovered `[2 -> 1]`—a hybrid combination not present in the 7 Oracle cascade routes (`02_generate_oracle.py` only had `[2 -> 0]` and `[3 -> 1]`). This boosted accuracy by **+4.5% (`44.0% -> 48.5%`)** while cutting energy by **-33.1% to -36.2% (`1,075 J -> 687–720 J`)**.



### Phase 8: Subquery State-Machine Masking & Pareto Sweep (`v8`)

* **Key Architectural Additions**:
1. **Subquery State-Machine Grammar Mask**: Implemented `get_valid_action_mask(state, device)` to enforce valid subquery transitions (`Retrieve {2,3} -> [Optional Reason {4,5}] -> Answer Subquery {0,1}`), eliminating mid-trajectory subquery failures (`[6 -> 3 -> 1 -> 1 -> 1]`).
2. **Root-Step Gradient Undilution**: Set `traj_norm = 1.0` at `step_i == 0` so root routing choices receive full gradient weight, preventing long trajectory chains from suppressing early decision updates.



# SLOT 1: Direct v8 Control (Isolates Subquery Mask + Root Gradient Fix)
sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01 \
  --joule_lambda 1.0e-7 \
  --gradient_accum 8 \
  --run_name "greenrag-grpo-v8-masked-5e6" \
  --output_dir "models/greenrag-grpo-v8-masked-5e6"

# SLOT 2: Pure Discriminative Accuracy (Zero All-Pass Penalty / Skips All-Pass Batches via std_r < 1e-6)
sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01 \
  --joule_lambda 0.0 \
  --gradient_accum 8 \
  --run_name "greenrag-grpo-v8-acc-only" \
  --output_dir "models/greenrag-grpo-v8-acc-only"

# SLOT 3: High-Thrift Pareto Point (Calibrated Linear Joule Penalty + Light KL Tether)
sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 5.0e-6 \
  --kl_coef 0.002 \
  --joule_lambda 5.0e-5 \
  --gradient_accum 8 \
  --run_name "greenrag-grpo-v8-high-thrift" \
  --output_dir "models/greenrag-grpo-v8-high-thrift"

  

* **Phase 8 Key Findings & Behavioral Dynamics**:
1. **`v8-acc-only` Unlocked True Multi-Hop Decomposition & Reasoning (75% Peak Eval Accuracy)**:
* Because `joule_lambda = 0.0` sets `std_r == 0.0 < 1e-6` on all-pass batches, easy single-hop questions are completely skipped (`Loss = 0.0, KL = 0.0`). This eliminated the "All-Pass KL Pull" that previously dragged policies back toward their single-hop SFT prior on easy questions.


* Consequently, `tools/decompose_pct` rose to **`0.10–0.14`**, with organic rollouts (`Gen 2–4`) actively selecting `DEC_SLM (6)` and `DEC_LLM (7)` and inserting `REASON_LLM (5)` before subquery/final generation (`[6 -> 3 -> 5 -> 1 -> 3 -> 5 -> 1 -> 1]` and `[3 -> 5 -> 1]`). Validation accuracy reached **`0.70` at Step 350**, **`0.75` at Step 400**, and **`0.70` at Step 500**.




2. **`v8-masked-5e6` Confirmed the All-Pass KL Tether Effect**:
* With `joule_lambda = 1.0e-7`, passing 2-step rollouts (`R = 1.000`) and passing 7-step Decompose rollouts (`R = 0.999`) have `std_r ≈ 1.4e-4 > 1e-6`. Thus, all-pass batches were not skipped, and the `0.01` KL penalty continuously anchored organic rollouts to `[2 -> 1]` (`RET_KEY -> GEN_LLM`).


* The run was rock-solid across all 1,000 steps (`0.55` eval accuracy at Step 1000, zero timeouts, `decompose_pct` locked at `0.0714–0.0833` corresponding to 1 forced Decompose + 3 two-step single-hop rollouts).




3. **`v8-high-thrift` Demonstrated the "Grammar Mask Lifeboat" Effect**:
* Lowering `kl_coef` to `0.002` under a heavy `5.0e-5` Joule penalty caused `KL` to spike to `4.1–5.5` between Steps 200 and 300. Because `get_valid_action_mask` only constrained trajectories *after* `Decompose` was invoked, unmasked non-decompose rollouts (`Gen 2–4`) fell into 15-step retrieval loops (`reward_mean = -0.2125`, `eval/accuracy = 0.00` at Step 300).


* Because forced `Decompose` (`Gen 1/4`) was protected by the Subquery Grammar Mask and always terminated cleanly (`-0.10` vs `-0.25` timeout), `Decompose` accumulated a massive positive advantage during the crash. At Step 305, the policy fled into `Decompose` (`decompose_pct` jumped from `0.018` to `0.138`), rescuing `eval/accuracy` from **`0.00` (Step 300) back to `0.50` (Step 450)** before gradually annealing back to ultra-cheap single-hop (`KL = 0.22`) by Step 1000.

### Phase 9: Conditional KL Decoupling & Controlled $\lambda_J$ Sweep (`v9`)
- **Diagnosis of Phase 8 Bimodal Split**:
  - `v8-acc-only` (`joule_lambda = 0.0`) discovered emergent "Probe-Then-Decompose" routing (`[2 -> 6 -> ...]`) and reached **54.0% accuracy (+10.0% over SFT)**, but decomposed on 93.5%–100% of queries (`5,198–9,315 J`) due to zero energy penalty when single-hop and multi-hop both succeeded.
  - Conversely, `v8-masked-5e6` (`joule_lambda = 1e-7`) converged to 100% single-hop `[2 -> 1]` (`48.5%` @ `667.4 J`) because nonzero energy variance (`std_r > 1e-6`) kept all-pass batches active, causing the `kl_coef = 0.01` tether to pull the policy back toward the single-hop SFT prior on every easy question.
- **Key Modifications**:
  1. **Conditional KL Decoupling on All-Pass Batches (`05_train_grpo.py`)**: Set `effective_kl_coef = 0.0 if all_passed else kl_coef` inside `grpo_loss()`. When all rollouts succeed (`min(R) > 0`), the SFT KL pull is disabled while the Policy Gradient (`pg_loss`) remains active, allowing $\lambda_J$ to reward `[2 -> 1]` over `[2 -> 6 -> ...]` on easy queries without erasing multi-hop weights.
  2. **Upgraded Evaluation Telemetry (`06_eval_checkpoints.py`)**: Added subquery grammar masking (`get_valid_action_mask`), standard SQuAD/HotpotQA Exact Match (`EM`) and token-level `F1`, effective system power (`Watts = Joules / Seconds`), per-query JSON records, and multi-dataset (`--dataset`) support.

## Gemini's suggested values. Low: 5e-6, Med: 1.5e-5, High 3.5e-5
## My plan: Low: 1e-7, Med: 7e-7, High: 5e-6
## Med is roughly (very roughly the geometric mean (7.07))

- **Launch Commands (3-Slot Controlled $\lambda_J$ Sweep)**:
```bash
# SLOT 1: Light Energy Penalty (1.0e-7)
sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01 \
  --joule_lambda 1.0e-7 \
  --gradient_accum 8 \
  --run_name "greenrag-grpo-v9-cond-1e7" \
  --output_dir "models/greenrag-grpo-v9-cond-1e7"

# SLOT 2: Mid Energy Penalty (7e-7)
sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01 \
  --joule_lambda 7.0e-7 \
  --gradient_accum 8 \
  --run_name "greenrag-grpo-v9-cond-7e7" \
  --output_dir "models/greenrag-grpo-v9-cond-7e7"

# SLOT 3: High Energy Penalty (5e-6)
sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01 \
  --joule_lambda 5.0e-6 \
  --gradient_accum 8 \
  --run_name "greenrag-grpo-v9-cond-5e6" \
  --output_dir "models/greenrag-grpo-v9-cond-5e6"

  # SLOT A: 2.0e-5 (4x stronger than 5e6 — 5,000 J costs -0.10 reward)
sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01 \
  --joule_lambda 2.0e-5 \
  --gradient_accum 8 \
  --run_name "greenrag-grpo-v9-cond-2e5" \
  --output_dir "models/greenrag-grpo-v9-cond-2e5"

# SLOT B: 5.0e-5 (10x stronger than 5e6 — 5,000 J costs -0.25 reward)
sbatch hpc/run_05_train_grpo.sbatch \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01 \
  --joule_lambda 5.0e-5 \
  --gradient_accum 8 \
  --run_name "greenrag-grpo-v9-cond-5e5" \
  --output_dir "models/greenrag-grpo-v9-cond-5e5"

  # ------------------------------------------------------------------------------
# RUN 1: Resume v9-cond-7e7 from step_650 to step_1000
# ------------------------------------------------------------------------------
sbatch hpc/run_05_train_grpo.sbatch \
  --run_name greenrag-grpo-v9-cond-7e7-resumed \
  --output_dir models/greenrag-grpo-v9-cond-7e7 \
  --joule_lambda 7.0e-7 \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01 \
  --resume_from models/greenrag-grpo-v9-cond-7e7/step_650 \
  --start_step 650 \
  --total_steps 1500


# ------------------------------------------------------------------------------
# RUN 2: New intermediate lambda sweep (v9-cond-5e6)
# ------------------------------------------------------------------------------
sbatch hpc/run_05_train_grpo.sbatch \
  --run_name greenrag-grpo-v9-cond-5e6 \
  --output_dir models/greenrag-grpo-v9-cond-5e6 \
  --joule_lambda 5.0e-6 \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01


# ------------------------------------------------------------------------------
# RUN 3: New intermediate lambda sweep (v9-cond-2e6)
# ------------------------------------------------------------------------------
sbatch hpc/run_05_train_grpo.sbatch \
  --run_name greenrag-grpo-v9-cond-2e6 \
  --output_dir models/greenrag-grpo-v9-cond-2e6 \
  --joule_lambda 2.0e-6 \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01 \
  --total_steps 1000 \
  --datasets hotpot


# New plan: Continue training after the 1000 checkpoint hotpot training without trajectory forcing
# 
# 1. Highest Thrift (5.0e-5) — resumes from v9-cond-5e5/step_1000
# DONE to 1000 extra steps
sbatch hpc/run_05_train_grpo.sbatch \
  --run_name greenrag-grpo-v10-mixed-5e5 \
  --resume_from models/greenrag-grpo-v9-cond-5e5/step_1000 \
  --output_dir models/greenrag-grpo-v10-mixed-5e5 \
  --start_step 1000 \
  --total_steps 2000 \
  --joule_lambda 5.0e-5 \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01 \
  --datasets hotpot,squad,nq \
  --no-force-trajectory

# 2. Mid-High Thrift (2.0e-5) — resumes from v9-cond-2e5/step_1000
sbatch hpc/run_05_train_grpo.sbatch \
  --run_name greenrag-grpo-v10-mixed-2e5 \
  --resume_from models/greenrag-grpo-v9-cond-2e5/step_1000 \
  --output_dir models/greenrag-grpo-v10-mixed-2e5 \
  --start_step 1000 \
  --total_steps 2000 \
  --joule_lambda 2.0e-5 \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01 \
  --datasets hotpot,squad,nq \
  --no-force-trajectory

# 3. Balanced Thrift (5.0e-6) — resumes from v9-cond-5e6/step_1000
sbatch hpc/run_05_train_grpo.sbatch \
  --run_name greenrag-grpo-v10-mixed-5e6 \
  --resume_from models/greenrag-grpo-v9-cond-5e6/step_1000 \
  --output_dir models/greenrag-grpo-v10-mixed-5e6 \
  --start_step 1000 \
  --total_steps 2000 \
  --joule_lambda 5.0e-6 \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01 \
  --datasets hotpot,squad,nq \
  --no-force-trajectory

# 4. Mid-Low Penalty (2.0e-6) — change --resume_from to v9-cond-2e6/step_1000 once Stage 1 completes
sbatch hpc/run_05_train_grpo.sbatch \
  --run_name greenrag-grpo-v10-mixed-2e6 \
  --resume_from models/greenrag-grpo-v9-cond-5e6/step_1000 \
  --output_dir models/greenrag-grpo-v10-mixed-2e6 \
  --start_step 1000 \
  --total_steps 2000 \
  --joule_lambda 2.0e-6 \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01 \
  --datasets hotpot,squad,nq \
  --no-force-trajectory

# 5. Low Penalty (7.0e-7) — resumes from v9-cond-7e7/step_1000
sbatch hpc/run_05_train_grpo.sbatch \
  --run_name greenrag-grpo-v10-mixed-7e7 \
  --resume_from models/greenrag-grpo-v9-cond-7e7/step_1000 \
  --output_dir models/greenrag-grpo-v10-mixed-7e7 \
  --start_step 1000 \
  --total_steps 2000 \
  --joule_lambda 7.0e-7 \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01 \
  --datasets hotpot,squad,nq \
  --no-force-trajectory

  ## Phase 11 (v11): Bellman Causal Return-to-Go ($G_{i,t}$) & Depth-Aligned Baselines

### Motivation & Code Changes (`05_train_grpo.py`)
In **v9/v10**, `grpo_loss` broadcast a single full-trajectory return $G_{i,0}$ to every step $t$ in rollout $i$[cite: 1]. On multi-hop questions where 2 of 4 rollouts failed, successful trajectories with redundant keyword loops (`[2 -> 2 -> 2 -> 2 -> 1]`) rode their own accuracy coattails ($A_i = G_{i,0} - \bar{G}_0 > 0$), positively reinforcing redundant searches at $t \ge 2$ while adding sunk-cost variance ($J_{<t}$) to late-step updates[cite: 1].

To fix multi-step credit assignment without adding heuristic shaping penalties or step caps, **v11** updates `05_train_grpo.py` with three principled RL changes:
1. **Per-step Joule accounting (`step_joules`):** Records marginal hardware energy `max(0.0, curr_joules - prev_joules)` around every `engine.step(state, action_id)` call inside `rollout_one_trajectory`.
2. **Bellman causal return-to-go ($G_{i,t}$):** Replaces uniform trajectory return $G_{i,0}$ with step-specific causal returns $G_{i,t} = 1.0 - \lambda_J \sum_{k=t}^{T_i - 1} J_{i,k}$ for solved rollouts (keeping flat terminal penalties `-0.10` / `-0.25` on failures/timeouts to preserve the root anti-cowardice guarantee), stripping out sunk energy $J_{<t}$ spent before step $t$.
3. **Depth-aligned Dr. GRPO baselines ($A_{i,t} = G_{i,t} - \bar{G}_t$):** Computes step-level baselines at every depth $t$ across rollouts still active at step $t$ plus peers already absorbed into a solved terminal state ($G^{\text{abs}} = 1.0$), directly pitting early stopping (`GEN_SLM`) against additional retrieval (`RET_KEY`/`RET_VEC`) on remaining Joules.

### Hyperparameter Rationale
* **`--joule_lambda 2.0e-5`:** Calibrated Pareto knee from `v9-cond-2e5`[cite: 1]—each extra `~342 J` `RET_KEY` step costs `0.0068` return-to-go at depth $t$, giving early exit a clean positive advantage over redundant `>=4x` loops while keeping `5,000 J` decomposition (`R = 0.90`) viable on genuine multi-hop queries[cite: 1].
* **Learning Rate Sweep (`2.0e-6` vs. `5.0e-6`):**
  * **Run 1 (`2.0e-6`):** Lower step size allows smaller per-step marginal Joule differences ($\Delta G_t \approx 0.007\text{–}0.022$) to accumulate smoothly over `gradient_accum 8` without early batch variance locking in greedy `RET_KEY -> RET_KEY` transitions.
  * **Run 2 (`5.0e-6`):** Direct `v9` LR control[cite: 1] isolating the effect of causal $G_{i,t}$ against uniform $G_{i,0}$ under identical optimizer dynamics.

### Launch Commands

## MISLEADING, this was run as 2e5 in the name but with 2e6 as the value
```bash
# 1. Causal Return-to-Go (2.0e-5) — Stabilized LR (2.0e-6) from SFT
sbatch hpc/run_05_train_grpo.sbatch \
  --run_name greenrag-grpo-v11-causal-2e5-lr2e6 \
  --output_dir models/greenrag-grpo-v11-causal-2e5-lr2e6 \
  --start_step 0 \
  --total_steps 1000 \
  --joule_lambda 2.0e-6 \
  --learning_rate 2.0e-6 \
  --kl_coef 0.01 \
  --gradient_accum 8 \
  --datasets hotpot

# 2. Causal Return-to-Go (2.0e-5) — v9 Control LR (5.0e-6) from SFT
sbatch hpc/run_05_train_grpo.sbatch \
  --run_name greenrag-grpo-v11-causal-2e5-lr5e6 \
  --output_dir models/greenrag-grpo-v11-causal-2e5-lr5e6 \
  --start_step 0 \
  --total_steps 1000 \
  --joule_lambda 2.0e-6 \
  --learning_rate 5.0e-6 \
  --kl_coef 0.01 \
  --gradient_accum 8 \
  --datasets hotpot

  ## Phase 12 (v12): Optimizer/Trust-Region Recalibration, Standalone Pareto Eval & Mixed-Dataset Transfer

### Motivation & Experimental Design
In **Phase 11 (v11)**, Bellman causal return-to-go ($G_{i,t} = 1.0 - \lambda_J \sum_{k=t}^{T_i - 1} J_{i,k}$) and depth-aligned Dr. GRPO baselines ($A_{i,t} = G_{i,t} - \bar{G}_t$) eliminated mid-training pruning collapses on HotpotQA, sustaining 59.3% mean `eval/accuracy` on `v11-lr5e6` (holding 60.0%–65.0% from `step_100` through `step_650`) and 58.0% mean `eval/accuracy` on `v11-lr2e6` (peaking at 65.0% at `step_300`, `step_350`, and `step_700`)[cite: 1].

**Phase 12 (v12)** builds on those checkpoints in two stages:
1. **Stage 12A — Standalone $N=200$ Pareto Selection (`06_eval_checkpoints.py`):** Evaluates top `v11` checkpoints on the deterministic $N=200$ (`seed=42`) HotpotQA validation split to measure exact mean/median Joules, step counts, and action distributions against the `v9-cond-2e5` Pareto knee (`50.5%` accuracy @ `1,939.5 J`, `5.37` avg steps)[cite: 1].
2. **Stage 12B — Recalibrated Policy Optimization & Phase 2 Mixed-Dataset Transfer (`hotpot,squad,nq`, `--no-force-trajectory`):** Resumes from the top `v11` checkpoint(s) with tightened trust-region/optimizer controls and organic 4-rollout sampling (`forced_first_action = None`) across an equal mixture of multi-hop (`hotpot`) and single-hop (`squad`, `nq`) queries so the router learns single-hop escalation suppression without forgetting multi-hop decomposition[cite: 1].

### Code & Hyperparameter Recalibrations (`05_train_grpo.py`)
* **Learning Rate & Linear Warmup (`LambdaLR`):** Dropped peak `--learning_rate` from `5.0e-6` / `2.0e-6` to `1.0e-6` with a **40-step linear warmup** (`LambdaLR`) to prevent early optimizer shock when transitioning to organic mixed-dataset rollouts.
* **KL Coefficient & Tighter PPO Clipping:** Lowered `--kl_coef` from `0.01` to `0.0015` (with `0.001` as a secondary sweep point) so KL prevents vocabulary/logit drift without outbidding the energy reward signal, while tightening `CLIP_EPS` from `0.2` to `0.1` to bound single-step policy ratio updates.
* **Effective Batch Size (`--gradient_accum 8`):** Set `--gradient_accum 8` (up from the `4` default—`16` questions / `64` trajectories per optimizer step at `BATCH_SIZE = 2`, `NUM_GENERATIONS = 4`) to cut batch-level multi-hop variance in half[cite: 1].
* **Tighter Gradient Norm Clipping:** Applied `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)` immediately before `optimizer.step()` (tightened from `1.0`) to damp occasional high-variance multi-hop gradient spikes[cite: 1].
* **Phase 2 Workload & Rollout Mode:** Enabled `--datasets hotpot,squad,nq` (equal `1/3` sampling weights) and `--no-force-trajectory` (symmetric 4-rollout Dr. GRPO advantage normalization in `grpo_loss`)[cite: 1].

# Primary v12 Run: lr=1.0e-6 (40-step warmup), kl_coef=0.0015, clip_eps=0.1, grad_accum=8, max_norm=0.5
sbatch hpc/run_05_train_grpo.sbatch \
  --run_name greenrag-grpo-v12-mixed-2e5-lr1e6-kl15e4 \
  --output_dir models/greenrag-grpo-v12-mixed-2e5-lr1e6-kl15e4 \
  --total_steps 1000 \
  --joule_lambda 2.0e-5 \
  --learning_rate 1.0e-6 \
  --kl_coef 0.0015 \
  --gradient_accum 8 \
  --datasets hotpot,squad,nq \
  --no-force-trajectory

# Secondary KL Sensitivity Check: kl_coef=0.001
sbatch hpc/run_05_train_grpo.sbatch \
  --run_name greenrag-grpo-v12-mixed-2e5-lr1e6-kl1e3 \
  --output_dir models/greenrag-grpo-v12-mixed-2e5-lr1e6-kl1e3 \
  --start_step 650 \
  --total_steps 1000 \
  --joule_lambda 2.0e-5 \
  --learning_rate 1.0e-6 \
  --kl_coef 0.001 \
  --gradient_accum 8 \
  --datasets hotpot,squad,nq \
  --no-force-trajectory