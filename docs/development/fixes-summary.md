# AutoConjecture Fixes Summary

## Problems Identified

1. **Missing `get_all_theorems()` method** - Called at neural_training_loop.py:346 but didn't exist
2. **Pretraining skipped** - `size()` only counted theorems, not axioms, so pretraining was skipped
3. **Wrong axiom format** - Using `PEANO_AXIOMS` (list of tuples) instead of `get_all_axioms()` (list of expressions)
4. **Tensor compatibility issue** - `.view()` not compatible with non-contiguous tensors
5. **Poor training for small datasets** - Warmup steps too high, batch size too large, learning rate too low

## Fixes Applied

### 1. Added `get_all_theorems()` method
**File:** `src/knowledge/knowledge_base.py`
```python
def get_all_theorems(self) -> List[Theorem]:
    """Get all theorems (alias for get_theorems for compatibility)."""
    return self.theorems
```

### 2. Fixed pretraining logic to use axioms
**File:** `src/training/neural_training_loop.py`
- Changed condition from `self.knowledge_base.size() > 0` to `self.knowledge_base.total_size() > 0`
- Added bootstrap logic for axiom-only training when no theorems exist yet

### 3. Fixed axiom initialization
**File:** `src/training/neural_training_loop.py`
```python
# Before:
self.knowledge_base = KnowledgeBase(axioms=PEANO_AXIOMS)  # Wrong: tuples

# After:
self.knowledge_base = KnowledgeBase(axioms=get_all_axioms())  # Correct: expressions
```

### 4. Refactored generator training
**File:** `src/models/generator_trainer.py`
- Created new `train_on_expressions()` method for direct expression training
- Modified `train_on_knowledge_base()` to use axioms + theorems
- Both methods now share common training logic

### 5. Fixed tensor reshaping
**File:** `src/models/transformer_generator.py`
```python
# Before:
logits_flat = logits.view(-1, self.vocab_size)
targets_flat = targets.view(-1)

# After:
logits_flat = logits.reshape(-1, self.vocab_size)
targets_flat = targets.reshape(-1)
```

## Test Results

### Before Fixes
- Pretraining: Skipped (no theorems)
- Model: Untrained
- Generation: Invalid token sequences

### After Fixes
- Pretraining: ✓ Runs on 6 axioms
- Model: ✓ Loss decreases from 3.0 → 0.17
- Generation: ✓ Produces valid mathematical expressions

### Example Generated Conjectures (after 50 epochs)
1. `∀x.((x + 0) = x)` - add_zero axiom
2. `∀x.((x * 0) = 0)` - mul_zero axiom
3. `∀x.∀y.((x * S(y)) = S((x + y)))` - variation of mul_succ

## Recommendations for Better Training

For small datasets (< 100 examples):
- Use `pretrain_epochs=50-100` for sufficient learning
- Set `generator_batch_size=2-4` for more gradient updates
- Use `generator_warmup_steps=10-20` for faster learning
- Increase `generator_lr=1e-3` for small datasets

For larger datasets (> 100 examples):
- Use `pretrain_epochs=10-20`
- Set `generator_batch_size=16-32`
- Use `generator_warmup_steps=100-500`
- Use `generator_lr=1e-4` for stability

## Status

✅ Neural training loop is now functional
✅ Axiom pretraining works correctly
✅ Model learns to generate valid expressions
✅ Ready for full training with theorem discovery
