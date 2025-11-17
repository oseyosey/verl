# MIA-Weighted GRPO Loss

## Overview

MIA-weighted GRPO loss enables prompt-level gradient weighting based on precomputed MIA (Membership Inference Attack) scores. This allows prompts that are more likely to be members to drive stronger RL updates, improving the MIA signal during training.

## Quick Start

### Option 1: Command-line flags

Add three configuration flags to your training script:

```bash
python3 -m verl.trainer.main_ppo \
    # ... your existing config ...
    algorithm.use_mia_weighted_loss=True \
    algorithm.mia_weighting_mode=quadratic \
    algorithm.mia_invert_weights=True \
    # ... rest of config ...
```

### Option 2: Config file

Update your config YAML file (`ppo_trainer.yaml` or `ppo_megatron_trainer.yaml`):

```yaml
algorithm:
  # ... existing config ...
  use_mia_weighted_loss: True
  mia_weighting_mode: quadratic
  mia_invert_weights: True
```

## Configuration

These parameters are defined in `verl/trainer/config/ppo_trainer.yaml` (and `ppo_megatron_trainer.yaml`) and can be overridden via command-line or config file:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm.use_mia_weighted_loss` | bool | `False` | Enable MIA-weighted loss |
| `algorithm.mia_weighting_mode` | str | `"linear"` | Weight transformation: `"linear"`, `"quadratic"`, `"exponential"` |
| `algorithm.mia_invert_weights` | bool | `False` | Invert weights before transformation (use `1 - w`) |

### Recommended Settings

For MIA optimization where **lower MIA scores indicate higher membership likelihood**:

```bash
algorithm.use_mia_weighted_loss=True
algorithm.mia_weighting_mode=quadratic
algorithm.mia_invert_weights=True
```

**Why quadratic?** Creates stronger gradient separation between members and non-members:
- Linear mode: 9× gradient ratio (0.9 / 0.1)
- Quadratic mode: 81× gradient ratio (0.81 / 0.01)

## Data Requirements

Your dataset must include MIA weights in the `extra_info` field:

```python
{
    "prompt": "...",
    "ground_truth": "...",
    "extra_info": {
        "mia_weight": 0.85,  # Precomputed MIA score in [0, 1]
    }
}
```

MIA weights should be precomputed using your chosen MIA metric (e.g., min-k++, loss-based) and stored in your dataset's parquet files.

## How It Works

### The Key Insight

❌ **Incorrect approach** (doesn't work):
```python
# Multiplying rewards by MIA weights before GRPO normalization
rewards = rewards * mia_weights
advantages = grpo_normalize(rewards)  # Effect cancels out!
```

✅ **Correct approach** (our implementation):
```python
# Compute advantages with standard GRPO normalization
advantages = grpo_normalize(rewards)  # No MIA weights here

# Apply MIA weights to the loss AFTER advantage computation
loss = compute_loss(advantages) * mia_weights  # Weights applied here!
```

### Implementation Flow

1. **Advantage Computation**: GRPO computes advantages using standard per-prompt mean/std normalization
2. **Weight Extraction**: MIA weights are extracted from `data.non_tensor_batch["mia_weight"]`
3. **Weight Transformation**: Weights are transformed based on config mode
4. **Loss Weighting**: Per-token loss matrix is multiplied by prompt weights before aggregation
5. **Gradient Update**: High-weight prompts contribute more strongly to gradients

### Weight Transformations

```python
# Linear (default)
w(x) = mia_weight

# Quadratic (recommended)
w(x) = mia_weight²

# Exponential
w(x) = exp(mia_weight - 1)

# With invert=True, applied before transformation
w(x) = transform(1 - mia_weight)
```

## Example: Updating Your Training Script

```bash
#!/bin/bash
# Before (standard GRPO)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/path/to/train.parquet \
    # ... other config ...

# After (MIA-weighted GRPO)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_mia_weighted_loss=True \
    algorithm.mia_weighting_mode=quadratic \
    algorithm.mia_invert_weights=True \
    data.train_files=/path/to/train.parquet \
    # ... other config ...
```

## Monitoring

When MIA-weighted loss is enabled, check logs for:

```
MIA-weighted loss enabled: mode=quadratic, invert=True, 
weight_mean=0.7234, weight_std=0.1523, weight_min=0.2341, weight_max=0.9876
```

These statistics help verify:
- Weights are being loaded correctly
- Transformation is working as expected
- There's reasonable variance in weights (std > 0)

## Implementation Details

### Files Modified

1. **`verl/trainer/ppo/core_algos.py`**
   - `transform_mia_weights()`: Weight transformation function
   - `agg_loss()`: Applies prompt weights to loss matrix
   - `compute_policy_loss()`: Updated to accept `prompt_weights` parameter

2. **`verl/trainer/ppo/ray_trainer.py`**
   - `compute_advantage()`: Extracts and transforms MIA weights

3. **`verl/workers/actor/dp_actor.py`**
   - `update_policy()`: Passes prompt weights to loss functions

4. **`verl/workers/actor/megatron_actor.py`**
   - `forward_backward_batch()`: Passes prompt weights to loss functions

### Backward Compatibility

✅ The implementation is fully backward compatible:
- When `use_mia_weighted_loss=False` (default): Standard GRPO behavior
- When MIA weights not in data: Standard GRPO behavior
- Existing training scripts work without modification

## Expected Behavior

With MIA-weighted loss enabled:

| Prompt Type | MIA Weight | Gradient Contribution | Effect |
|-------------|------------|----------------------|--------|
| High-member | High (0.9) | Large | Model focuses on reconstructing |
| Low-member | Low (0.1) | Small | Model focuses less on reconstructing |

This creates a stronger separation in the MIA signal compared to standard GRPO.

## Troubleshooting

### Issue: No MIA weights found in logs

**Cause**: MIA weights not in dataset or config not enabled

**Solution**:
1. Verify dataset includes `mia_weight` in `extra_info`
2. Confirm `algorithm.use_mia_weighted_loss=True` is set
3. Check reward manager passes through `extra_info` fields

### Issue: All weights are identical (std=0)

**Cause**: MIA weight computation produces uniform scores

**Solution**:
1. Review MIA weight computation pipeline
2. Verify weights have variance across dataset
3. Check data loading process

### Issue: Training identical to standard GRPO

**Cause**: Weights might all be ≈1.0 with linear mode

**Solution**:
1. Check logged weight statistics (mean, std, min, max)
2. Try `quadratic` or `exponential` mode for stronger effect
3. Verify `invert_weights` setting matches your MIA score convention

## Testing

Run the test suite to verify implementation:

```bash
cd /gpfs/scrubbed/osey/Dataset_Distillation/DDRL
python tests/test_mia_weighted_loss.py
```

Expected output: `🎉 ALL TESTS PASSED! 🎉`

## Performance Impact

- **Memory overhead**: O(batch_size) tensor (negligible)
- **Computation overhead**: One element-wise multiplication per batch (negligible)
- **Training speed**: <1% impact when enabled, 0% when disabled

## References

- GRPO paper: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- Implementation discussion: See `verl/trainer/ppo/core_algos.py` for details

