"""
Tests for Phase 2 neural components.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.tokenizer import ExpressionTokenizer
from src.models.transformer_generator import TransformerGenerator
from src.generation.neural_generator import NeuralConjectureGenerator
from src.models.curriculum import CurriculumScheduler, CurriculumConfig
from src.logic.terms import Var, Zero, Succ, Add, Mul
from src.logic.expressions import Equation, Forall


class TestExpressionTokenizer:
    """Test expression tokenizer."""

    def test_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = ExpressionTokenizer(max_length=128)
        assert tokenizer.vocab_size > 0
        assert tokenizer.pad_id == 0
        assert tokenizer.sos_id == 1
        assert tokenizer.eos_id == 2

    def test_encode_simple_equation(self):
        """Test encoding a simple equation."""
        tokenizer = ExpressionTokenizer()

        # Create simple equation: x = 0
        eq = Equation(Var("x"), Zero())
        tokens = tokenizer.encode_expression(eq)

        assert len(tokens) > 0
        assert tokens[0] == tokenizer.sos_id
        assert tokens[-1] == tokenizer.eos_id

    def test_encode_forall(self):
        """Test encoding a universally quantified expression."""
        tokenizer = ExpressionTokenizer()

        # Create: forall x. x = x
        eq = Equation(Var("x"), Var("x"))
        forall_expr = Forall(Var("x"), eq)
        tokens = tokenizer.encode_expression(forall_expr)

        assert len(tokens) > 0
        assert tokens[0] == tokenizer.sos_id
        assert tokens[-1] == tokenizer.eos_id

    def test_encode_decode_roundtrip(self):
        """Test that encoding and decoding produces the same expression."""
        tokenizer = ExpressionTokenizer()

        # Create expression: 0 = 0
        original = Equation(Zero(), Zero())
        tokens = tokenizer.encode_expression(original)
        decoded = tokenizer.decode_tokens(tokens)

        assert decoded is not None
        assert str(decoded) == str(original)

    def test_encode_complex_expression(self):
        """Test encoding a more complex expression."""
        tokenizer = ExpressionTokenizer()

        # Create: (x + 0) = x
        left = Add(Var("x"), Zero())
        right = Var("x")
        eq = Equation(left, right)
        tokens = tokenizer.encode_expression(eq)

        assert len(tokens) > 0
        decoded = tokenizer.decode_tokens(tokens)
        assert decoded is not None

    def test_pad_sequence(self):
        """Test sequence padding."""
        tokenizer = ExpressionTokenizer(max_length=20)

        tokens = [1, 2, 3, 4, 5]
        padded = tokenizer.pad_sequence(tokens, max_len=10)

        assert len(padded) == 10
        assert padded[:5] == tokens
        assert all(t == tokenizer.pad_id for t in padded[5:])

    def test_batch_encode(self):
        """Test batch encoding."""
        tokenizer = ExpressionTokenizer()

        expressions = [
            Equation(Zero(), Zero()),
            Equation(Var("x"), Zero()),
            Equation(Var("x"), Var("x"))
        ]

        batch = tokenizer.batch_encode(expressions, pad=True)

        assert len(batch) == 3
        assert all(len(seq) == len(batch[0]) for seq in batch)


class TestTransformerGenerator:
    """Test transformer generator model."""

    def test_initialization(self):
        """Test model initialization."""
        model = TransformerGenerator(
            vocab_size=50,
            d_model=64,
            nhead=4,
            num_layers=2
        )

        assert model.vocab_size == 50
        assert model.d_model == 64

    def test_forward_pass(self):
        """Test forward pass."""
        model = TransformerGenerator(
            vocab_size=50,
            d_model=64,
            nhead=4,
            num_layers=2
        )

        # Create dummy input
        batch_size = 2
        seq_len = 10
        src = torch.empty(0, batch_size, dtype=torch.long)
        tgt = torch.randint(0, 50, (seq_len, batch_size))

        # Forward pass
        logits = model(src, tgt)

        assert logits.shape == (seq_len, batch_size, 50)

    def test_generate(self):
        """Test sequence generation."""
        model = TransformerGenerator(
            vocab_size=50,
            d_model=64,
            nhead=4,
            num_layers=2
        )

        # Generate sequences
        sequences = model.generate(
            batch_size=3,
            max_length=20,
            temperature=1.0,
            sos_token_id=1,
            eos_token_id=2,
            device="cpu"
        )

        assert sequences.shape[0] == 3
        assert sequences.shape[1] <= 20

    def test_compute_loss(self):
        """Test loss computation."""
        model = TransformerGenerator(
            vocab_size=50,
            d_model=64,
            nhead=4,
            num_layers=2
        )

        seq_len = 10
        batch_size = 2
        logits = torch.randn(seq_len, batch_size, 50)
        targets = torch.randint(0, 50, (seq_len, batch_size))

        loss = model.compute_loss(logits, targets)

        assert loss.item() > 0


class TestNeuralConjectureGenerator:
    """Test neural conjecture generator."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = NeuralConjectureGenerator(
            device="cpu",
            max_length=64
        )

        assert generator.model is not None
        assert generator.tokenizer is not None

    def test_generate_conjectures(self):
        """Test conjecture generation."""
        generator = NeuralConjectureGenerator(
            device="cpu",
            max_length=64
        )

        # Generate some conjectures
        conjectures = generator.generate(n=5)

        assert len(conjectures) == 5
        # Some might be None (invalid), that's ok for untrained model

    def test_save_load(self, tmp_path):
        """Test saving and loading generator."""
        generator = NeuralConjectureGenerator(
            device="cpu",
            max_length=64
        )

        # Save
        save_path = tmp_path / "generator.pt"
        generator.save(str(save_path))

        # Load
        loaded = NeuralConjectureGenerator.load(str(save_path), device="cpu")

        assert loaded.tokenizer.vocab_size == generator.tokenizer.vocab_size
        assert loaded.max_length == generator.max_length

    def test_temperature_setting(self):
        """Test temperature setting."""
        generator = NeuralConjectureGenerator(device="cpu")

        generator.set_temperature(2.0)
        assert generator.temperature == 2.0

        generator.set_top_k(100)
        assert generator.top_k == 100


class TestCurriculumScheduler:
    """Test curriculum learning scheduler."""

    def test_initialization(self):
        """Test scheduler initialization."""
        config = CurriculumConfig(
            initial_complexity=2,
            final_complexity=15
        )
        scheduler = CurriculumScheduler(config)

        assert scheduler.current_complexity == 2
        assert scheduler.current_stage == 0

    def test_complexity_range(self):
        """Test complexity range calculation."""
        config = CurriculumConfig(initial_complexity=5)
        scheduler = CurriculumScheduler(config)

        min_c, max_c = scheduler.get_current_complexity_range()
        assert min_c <= scheduler.current_complexity
        assert max_c >= scheduler.current_complexity

    def test_record_results(self):
        """Test recording results."""
        config = CurriculumConfig(min_samples_per_stage=10)
        scheduler = CurriculumScheduler(config)

        for i in range(10):
            scheduler.record_result(success=True, complexity=5)

        assert scheduler.stage_samples == 10
        assert scheduler.stage_successes == 10

    def test_stage_advancement(self):
        """Test stage advancement."""
        config = CurriculumConfig(
            initial_complexity=2,
            min_samples_per_stage=10,
            success_threshold=0.5
        )
        scheduler = CurriculumScheduler(config)

        # Record enough successful results
        for i in range(15):
            scheduler.record_result(success=True, complexity=2)

        assert scheduler.should_advance_stage()

        initial_complexity = scheduler.current_complexity
        scheduler.advance_stage()

        assert scheduler.current_complexity > initial_complexity
        assert scheduler.current_stage == 1
        assert scheduler.stage_samples == 0  # Reset after advance

    def test_temperature_scheduling(self):
        """Test temperature scheduling."""
        config = CurriculumConfig(
            initial_complexity=2,
            final_complexity=10,
            initial_temperature=1.5,
            final_temperature=0.8
        )
        scheduler = CurriculumScheduler(config)

        initial_temp = scheduler.get_current_temperature()
        assert initial_temp == config.initial_temperature

        # Advance several stages
        for _ in range(3):
            scheduler.advance_stage()

        later_temp = scheduler.get_current_temperature()
        assert later_temp < initial_temp

    def test_filter_by_complexity(self):
        """Test filtering expressions by complexity."""
        config = CurriculumConfig(initial_complexity=5)
        scheduler = CurriculumScheduler(config)

        # Create expressions with different complexities
        expressions = [
            Equation(Zero(), Zero()),  # Low complexity
            Equation(Var("x"), Var("x")),  # Low complexity
            Equation(Add(Var("x"), Zero()), Var("x")),  # Medium complexity
        ]

        # This is a simple test - actual filtering depends on complexity calculation
        filtered = scheduler.filter_by_complexity(expressions)
        assert isinstance(filtered, list)


class TestIntegration:
    """Integration tests for Phase 2 components."""

    def test_tokenizer_with_generator(self):
        """Test tokenizer integration with generator."""
        tokenizer = ExpressionTokenizer(max_length=64)
        model = TransformerGenerator(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            nhead=4,
            num_layers=2
        )

        generator = NeuralConjectureGenerator(
            model=model,
            tokenizer=tokenizer,
            device="cpu"
        )

        # Should be able to generate without errors
        conjectures = generator.generate(n=3)
        assert len(conjectures) == 3

    def test_curriculum_with_expressions(self):
        """Test curriculum with actual expressions."""
        config = CurriculumConfig(
            initial_complexity=2,
            final_complexity=10,
            min_samples_per_stage=5
        )
        scheduler = CurriculumScheduler(config)

        # Create some expressions
        expressions = [
            Equation(Zero(), Zero()),
            Equation(Var("x"), Zero()),
            Forall(Var("x"), Equation(Var("x"), Var("x")))
        ]

        # Test filtering
        filtered = scheduler.filter_by_complexity(expressions)
        assert isinstance(filtered, list)

        # Test sampling
        sampled = scheduler.sample_by_stage(expressions, n_samples=2)
        assert len(sampled) <= 2


def test_phase2_imports():
    """Test that all Phase 2 components can be imported."""
    from src.models.tokenizer import ExpressionTokenizer
    from src.models.transformer_generator import TransformerGenerator
    from src.generation.neural_generator import NeuralConjectureGenerator
    from src.models.generator_trainer import GeneratorTrainer, GeneratorTrainingConfig
    from src.models.curriculum import CurriculumScheduler, CurriculumConfig
    from src.training.neural_training_loop import NeuralTrainingLoop, NeuralTrainingConfig

    assert ExpressionTokenizer is not None
    assert TransformerGenerator is not None
    assert NeuralConjectureGenerator is not None
    assert GeneratorTrainer is not None
    assert CurriculumScheduler is not None
    assert NeuralTrainingLoop is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
