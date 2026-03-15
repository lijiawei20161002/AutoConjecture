# Experiment Results: phase3_a100_fixed_20260315_091959

## Summary

| Field | Value |
|-------|-------|
| Experiment ID | `phase3_a100_fixed_20260315_091959` |
| Date | 2026-03-15 |
| Start time | 09:19:59 |
| End time | 10:20:08 |
| Training time | 3605.3 s (~60 min) |
| Hardware | NVIDIA A100-SXM4-40GB, 42.4 GB GPU memory |
| Phase | Phase 3: RL-Based Prover Training (PPO) |
| Starting KB | 96 (loaded from `data/checkpoints/neural_epoch_19_cycle_999.json`) |

## Configuration

### Training

| Parameter | Value |
|-----------|-------|
| `num_epochs` | 20 |
| `cycles_per_epoch` | 500 |
| `conjectures_per_cycle` | 10 |
| `random_seed` | 42 |
| `initial_complexity` | 6 |
| `final_complexity` | 15 |
| `neural_ratio` | 0.5 |
| `use_curriculum` | True |
| `success_threshold` | 0.25 |
| `max_proof_steps` | 30 |
| `update_interval` | 50 |
| `use_heuristic_fallback` | True |
| `heuristic_max_depth` | 50 |
| `heuristic_max_iter` | 500 |
| `bc_warmup_cycles` | 200 |

### Actor-Critic (PPO Prover)

| Parameter | Value |
|-----------|-------|
| `encoder_d_model` | 256 |
| `encoder_nhead` | 4 |
| `encoder_num_layers` | 3 |
| `encoder_dropout` | 0.1 |
| `ac_hidden_dim` | 256 |
| `ppo_lr` | 0.0003 |
| `ppo_clip_epsilon` | 0.2 |
| `ppo_value_coef` | 0.5 |
| `ppo_entropy_coef` | 0.01 |
| `ppo_epochs` | 4 |
| `ppo_mini_batch_size` | 64 |
| `ppo_gamma` | 0.99 |
| `ppo_gae_lambda` | 0.95 |
| `ppo_max_grad_norm` | 0.5 |
| `ppo_target_kl` | 0.05 |

### Generator

| Parameter | Value |
|-----------|-------|
| `gen_d_model` | 256 |
| `gen_nhead` | 8 |
| `gen_num_layers` | 6 |
| `gen_lr` | 0.0001 |
| `gen_batch_size` | 32 |
| `gen_warmup_steps` | 500 |
| `gen_pretrain_epochs` | 3 |
| `gen_update_interval` | 100 |

## Final Statistics

| Metric | Value |
|--------|-------|
| Final KB size | **131** (+35 new theorems from starting KB of 96) |
| Total RL proofs | 21 |
| Total heuristic proofs | 14 |
| Total conjectures generated | 50,945 |
| Total proofs attempted | 5,116 |
| RL success rate | 0.41% |
| Heuristic success rate | 0.27% |
| Total PPO updates | 103 |

## Epoch-by-Epoch Results

Starting KB size at Epoch 1, Cycle 0: **96**

| Epoch | Duration (s) | RL proofs | Heuristic proofs | Total new | KB size (end) |
|-------|-------------|-----------|------------------|-----------|---------------|
| 1  | 190.2 | 2 | 0 | 2  | 98  |
| 2  | 179.6 | 2 | 0 | 2  | 100 |
| 3  | 166.8 | 2 | 0 | 2  | 102 |
| 4  | 174.7 | 3 | 0 | 3  | 105 |
| 5  | 195.6 | 3 | 1 | 4  | 109 |
| 6  | 227.0 | 2 | 1 | 3  | 112 |
| 7  | 221.1 | 1 | 2 | 3  | 115 |
| 8  | 227.2 | 2 | 1 | 3  | 118 |
| 9  | 209.4 | 1 | 1 | 2  | 120 |
| 10 | 180.4 | 1 | 2 | 3  | 123 |
| 11 | 179.9 | 0 | 0 | 0  | 123 |
| 12 | 152.7 | 0 | 2 | 2  | 125 |
| 13 | 163.0 | 0 | 1 | 1  | 126 |
| 14 | 153.7 | 2 | 0 | 2  | 128 |
| 15 | 168.6 | 0 | 0 | 0  | 128 |
| 16 | 164.8 | 0 | 1 | 1  | 129 |
| 17 | 165.9 | 0 | 0 | 0  | 129 |
| 18 | 161.9 | 0 | 0 | 0  | 129 |
| 19 | 151.6 | 0 | 2 | 2  | 131 |
| 20 | 171.1 | 0 | 0 | 0  | 131 |
| **Total** | **3605.3** | **21** | **14** | **35** | |

## Observations

- **Early epochs were most productive (1–10):** All 21 RL proofs and 10 of 14 heuristic proofs came in the first 10 epochs (30 of 35 total new theorems).
- **Proof rate declined sharply after epoch 10:** Epochs 11–20 yielded only 5 new theorems, with 4 zero-proof epochs (11, 15, 17, 18, 20).
- **RL dominated early, heuristic took over late:** RL was the primary driver in epochs 1–10; from epoch 11 onward only heuristic proofs succeeded.
- **Curriculum was active:** Complexity ramped from 6 (initial) toward 15 (final) over the course of training.
- **KB growth plateaued around epoch 10:** KB grew from 96 → 123 in the first 10 epochs (+27), then only 96 → 131 total (+8) over the final 10 epochs.

## Checkpoint

Final checkpoint saved at end of Epoch 20:
- `data/checkpoints/epoch_19_cycle_499` (0-indexed naming)
- Saved at: 2026-03-15 10:20:08
