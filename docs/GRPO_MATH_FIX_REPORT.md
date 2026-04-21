# GRPO Mathematical and Architectural Fixes Report

## 1. PPO Policy Gradient Surrogate Ratio Fix
**Issue**: The previous `grpo_loss` calculation mathematically skewed the Trust Region by using the reference policy (`ref_logp`) to normalize the current action probabilities (`ratio = exp(chosen_logp - ref_logp)`). Since the reference policy is a fixed SFT baseline, this violated PPO's clipped surrogate formulation.
**Fix**: Adjusted the formula to properly isolate the active model by detaching the initial forward pass probability. The ratio is now correctly anchored to the previous step in the active trajectory:
```python
old_logp = chosen_logp.detach()
ratio = torch.exp(chosen_logp - old_logp)  # PPO ratio w.r.t OLD policy, not reference!
```

## 2. Backward Pass Aggregation (Graph Execution)
**Issue**: Loss `.backward()` was being executed locally inside of individual trajectory loops across the generated group. This caused gradients to isolate per trajectory, leading to invalid accumulations and PyTorch computation graph tearing.
**Fix**: Extracted the backpropagation call out of the generation loop. Trajectory-level losses are now accumulated dynamically as tensors into a central `total_loss` pool. `final_loss.backward()` is called singularly across the entire group batch, perfectly aligning with standard GRPO architecture.

## 3. KL Divergence Formula Persistence
**Insight**: An external analysis suggested simplifying the KL Divergence penalty to standard PPO approximations. 
**Decision**: Maintained the DeepSeek exact unbiased KL estimator (`exp(-kl_log_ratio) + kl_log_ratio - 1.0`). This accurately reflects the true divergence properties and prevents the policy gradient trust boundaries from artificially accelerating divergence compared to simple approximations.