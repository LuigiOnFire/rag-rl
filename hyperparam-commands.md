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