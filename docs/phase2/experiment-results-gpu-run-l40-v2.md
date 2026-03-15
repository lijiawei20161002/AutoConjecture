# Experiment Results: gpu_run_l40_v2

## Summary

| Field | Value |
|-------|-------|
| Experiment ID | `gpu_run_l40_v2` |
| Date | 2026-03-15 |
| Start time | 04:26:06 |
| End time | 06:46:35 |
| Training time | 8426.23 s (2h 20m 26s) |
| Hardware | NVIDIA L40 GPU (`device=cuda`) |
| Phase | Phase 2: Neural Generator Training |

## Model Configuration

| Parameter | Value |
|-----------|-------|
| `d_model` | 512 |
| `nhead` | 8 |
| `num_layers` | 8 |
| `dropout` | 0.1 |
| `generator_lr` | 0.0001 |
| `generator_batch_size` | 32 |
| `generator_warmup_steps` | 500 |
| `num_epochs` | 20 |
| `cycles_per_epoch` | 1000 |
| `conjectures_per_cycle` | 20 |
| `max_proof_depth` | 50 |
| `max_proof_iterations` | 1000 |
| `pretrain_epochs` | 5 |
| `mixed_generation` | True |
| `neural_ratio` | 0.3 |
| `use_curriculum` | False |
| `random_seed` | 42 |
| Model parameters | 25,252,881 |

The run resumed from a mid-training checkpoint (`trainer_epoch_9_cycle_500.pt`) from an earlier run.

## Axiom System (Peano Arithmetic fragment)

```
∀x.¬(S(x) = 0)
∀x.∀y.((S(x) = S(y)) → (x = y))
∀x.((x + 0) = x)
∀x.∀y.((x + S(y)) = S((x + y)))
∀x.((x * 0) = 0)
∀x.∀y.((x * S(y)) = ((x * y) + x))
```

## Final Statistics

| Metric | Value |
|--------|-------|
| Total proofs found | **865** |
| Conjectures generated | 289,239 |
| Proofs attempted | 52,097 |
| Overall success rate | **1.66%** |
| Average theorem complexity | **20.46** |
| Knowledge base size (final) | 865 |

## Epoch-by-Epoch Proof Counts

The log uses 1-indexed epoch labels; checkpoint filenames use 0-indexed (log "Epoch N" → `neural_epoch_{N-1}_cycle_500.json`).

| Log epoch | Internal epoch | Proofs this epoch | Cumulative KB size |
|-----------|---------------|-------------------|--------------------|
| 1 | 0 | 43 | 43 |
| 2 | 1 | 51 | 94 |
| 3 | 2 | 51 | 145 |
| 4 | 3 | 40 | 185 |
| 5 | 4 | 46 | 231 |
| 6 | 5 | 47 | 278 |
| 7 | 6 | 38 | 316 |
| 8 | 7 | 37 | 353 |
| 9 | 8 | 38 | 391 |
| 10 | 9 | 51 | 442 |
| 11 | 10 | 38 | 480 |
| 12 | 11 | 43 | 523 |
| 13 | 12 | 43 | 566 |
| 14 | 13 | 51 | 617 |
| 15 | 14 | 49 | 666 |
| 16 | 15 | 43 | 709 |
| 17 | 16 | 39 | 748 |
| 18 | 17 | 38 | 786 |
| 19 | 18 | 38 | 824 |
| 20 | 19 | 41 | **865** |
| **Total** | | **865** | |

Epoch sequence: 43, 51, 51, 40, 46, 47, 38, 37, 38, 51, 38, 43, 43, 51, 49, 43, 39, 38, 38, 41

## Complexity Distribution

Complexity is measured as AST node count of the theorem statement.

| Range | Count | Percentage |
|-------|-------|------------|
| < 10 | 11 | 1.3% |
| 10–15 | 70 | 8.1% |
| 15–20 | 267 | 30.9% |
| 20–25 | 355 | 41.0% |
| 25–30 | 147 | 17.0% |
| 30+ | 15 | 1.7% |
| **Total** | **865** | |

- Min complexity: 8.0
- Max complexity: 33.0
- Mean complexity: 20.46

The distribution is unimodal, peaking in the 20–25 range. Very low-complexity theorems (trivial identities like `((0 * 0) = 0)`) are rare because they are rapidly discovered and filtered as "already known" in later epochs.

## Sample Theorems by Complexity Range

**Complexity < 10** (trivial identities):
```
((0 * x) = 0)          [complexity 9]
((0 * 0) = 0)          [complexity 8]
(z = (z + 0))          [complexity 9]
```

**Complexity 10–15** (simple arithmetic):
```
(0 = (0 * S((0 + z))))          [complexity 14]
((z * (0 + 0)) = 0)             [complexity 12]
((0 * (0 + S(0))) = 0)          [complexity 13]
```

**Complexity 15–20** (moderate length):
```
((0 * ((0 + 0) + w)) = 0)                            [complexity 15]
(0 = (S(S(w)) * (w * 0)))                            [complexity 15]
((0 + 0) = ((0 * (0 + 0)) * S(0)))                   [complexity 17]
```

**Complexity 20–25** (typical output, most common range):
```
(((S(0) + (0 + 0)) * 0) = (((y + 0) * 0) + 0))            [complexity 22]
(((x * (0 + 0)) * (w + (x * 0))) = (0 + (w * 0)))          [complexity 24]
(((0 * 0) + ((0 + 0) * 0)) = ((0 * (x + 0)) * 0))          [complexity 23]
```

**Complexity 25–30** (complex):
```
((((x + 0) + y) * ((z + 0) * (w * 0))) = (y * 0))                        [complexity 26]
((S(0) * 0) = (((y * 0) * (z + 0)) + ((0 * 0) * 0)))                     [complexity 25]
((((0 + z) * 0) * S(0)) = (((x * w) * 0) * ((0 + 0) * S(0))))            [complexity 29]
```

**Complexity 30+** (most complex, rare):
```
((((0 + 0) * (0 * z)) + ((z * y) * (w * 0))) = (0 * (S(x) * S(x))))     [complexity 32]
(((0 * S(0)) + 0) = (((x + w) * (0 + z)) * ((y * 0) * (0 + 0))))        [complexity 31]
(((S(z) * 0) * (0 * S(w))) = (((0 * z) * (x + y)) * S((x * 0))))        [complexity 31]
```

## Checkpoint Files

| File | Theorems | Saved at | Notes |
|------|----------|----------|-------|
| `neural_epoch_0_cycle_500.json` | 40 | 04:29:02 | Log epoch 1 midpoint |
| `neural_epoch_1_cycle_500.json` | – | – | |
| ... | ... | ... | |
| `neural_epoch_16_cycle_500.json` | – | – | |
| `neural_epoch_17_cycle_500.json` | 784 | 06:25:20 | **Largest original checkpoint** (log epoch 18 midpoint) |
| `neural_epoch_18_cycle_500.json` | 24 | 06:46:33 | Resume run (fresh KB, log epoch 19 midpoint) |
| `neural_epoch_19_cycle_500.json` | 86 | 06:51:57 | Resume run |
| `neural_epoch_19_cycle_999.json` | 96 | 06:55:26 | Resume run — **overwrote original final checkpoint** |
| `neural_epoch_19_cycle_1000_reconstructed.json` | **865** | reconstructed | See below |

## Reconstructed Knowledge Base

**File:** `data/checkpoints/neural_epoch_19_cycle_1000_reconstructed.json`

The original final checkpoint (`neural_epoch_19_cycle_999.json`, saved at 06:46:37) was overwritten ~9 minutes later by a resume run that began at 06:43:12, which used a separate, fresh 24-theorem KB. The resume run's final checkpoint contained only 96 theorems.

Reconstruction method:
1. Start from `neural_epoch_17_cycle_500.json` (784 theorems, saved at 06:25:20).
2. Extract all 81 remaining theorem proofs from `run.log` lines after the checkpoint save at line 125407 (timestamp 06:25:21).
3. The log records every proved theorem with its statement, proof length, and complexity score via `✓ Proved:` entries.
4. Assign epoch numbers by timestamp boundaries: epoch 17 (2 theorems, 06:25–06:29:26), epoch 18 (38 theorems, 06:29:26–06:36:43), epoch 19 (41 theorems, 06:36:43–06:46:35).
5. No duplicates between the 784-theorem base and the 81 extracted theorems.

**Total recovered: 784 + 81 = 865 theorems**, matching the `run.log` final count exactly.

## Notes on the Resume Run

The original run completed successfully at 06:46:35 with 865 theorems. Approximately 9 minutes before the run finished (at 06:37:16), a resume attempt was launched from `trainer_epoch_18_cycle_500.pt` using CPU (`device=cpu`). This CPU attempt ran briefly then was re-launched at 06:43:12 on GPU (`device=cuda`). Both resume runs started from the mid-training checkpoint `neural_epoch_18_cycle_500.json`, which contained only 24 theorems (it was the resume run's own checkpoint, not the 821-theorem checkpoint from the original run at that epoch). When the GPU resume run saved its final checkpoint as `neural_epoch_19_cycle_999.json` at 06:55:26, it overwrote the original run's 865-theorem checkpoint that had been saved there at 06:46:37.

The original run's KB is fully recoverable from the log because all proved theorems are individually logged with statement, proof length, and complexity.
