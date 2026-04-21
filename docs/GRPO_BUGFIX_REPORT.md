# GRPO Training Bugfix Report

**Date**: April 15, 2026
**File**: `scripts/05_train_grpo.py`

This document details two critical bugs found and patched in the GRPO custom training loop that were causing the agent to experience severe performance decay and silent misfires.

---

## 1. The Zero-Variance "Unlearning" Catastrophe
**Symptoms:** 
The RL agent would successfully discover a good trajectory pattern (such as `RET` -> `GEN`), but instead of "locking it in" and consolidating the behavior, the model's performance would rapidly decay back to its baseline before oscillating.

**Root Cause:**
In GRPO, the advantage is calculated as `(reward - mean(rewards)) / std(rewards)`. If the agent uniformly succeeds across its entire rolling batch (or uniformly fails with exactly the same penalty), the standard deviation of rewards approaches `0`. 
When variance is zero, the calculated Advantage is exactly `0`. 
Consequently, the main Policy Gradient (PG) loss becomes `0`. However, the KL divergence penalty (which penalizes the policy for moving away from the reference SFT model) was **still being applied and computing active gradients**. 

This meant that whenever the network completely mastered a batch, it executed a massive KL-only backward pass that commanded the model to explicitly *unlearn* the successful RL pattern and revert to the frozen SFT weights. 

**The Fix:**
Implemented a variance check safeguard before calculating advantages. 
```python
if std_r < 1e-5:
    return 0.0, 0.0
```
If a batch produces zero variance, we now skip the update entirely. This explicitly prevents the KL penalty from destructively wiping out consolidated learning progress.

---

## 2. BPE String-Merging Bug (Phase 2 Token Misalignment)
**Symptoms:**
The agent struggled to learn the correct action distributions, appearing to fire blindly despite the codebase featuring advanced tensor-slicing logic (`[:prompt_len + 1]`) designed to cleanly isolate the action digit.

**Root Cause:**
In Phase 1, the LLM generates raw tokens based on a prompt. In Phase 2 (the backward pass), the original code was concatenating the `prompt` string and the `completion` string, and running that combined string back through the LLaMA/BPE tokenizer.

Because of how Byte-Pair Encoding works, `tokenizer("...\nAction:") + tokenizer(" 2\n")` does **not** securely map 1-to-1 to `tokenizer("...\nAction: 2\n")`. By appending text, the tokenizer would retroactively merge boundaries (e.g., merging the colon `:` and the space ` `, or the space ` ` and the `2`). 
Because the tokens merged differently than they did in the prompt, the length of the new sequence was shifted. The index `prompt_len` was no longer accurate, meaning the target token sliced by `input_ids[:, :prompt_len + 1]` was often missing the numeric action digit entirely (mistakenly optimizing for a newline or space instead).

**The Fix:**
Phase 2 now completely bypasses the tokenizer. `rollout_one_trajectory` physically caches the exact `int` mappings of the tensor `prompt_ids` and `new_token_ids` that the active LLM generated in Phase 1. 

Phase 2 was refactored to reconstitute this tensor directly via list concatenation:
```python
full_ids = prompt_ids + new_token_ids
```
This guarantees perfect 1-to-1 alignment of the tensor boundaries in Phase 2, ensuring the loss function is calculated exactly on the intended action ID.