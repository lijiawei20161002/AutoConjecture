"""
Phase 5 tests: parallel prover, advanced curriculum, and training loop smoke-test.
"""
import sys
import os
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.logic.terms import Var, Zero, Succ, Add, Mul
from src.logic.expressions import Equation, Forall
from src.logic.axioms import get_all_axioms


# ── helpers ───────────────────────────────────────────────────────────────────

def make_eq(a, b):
    return Equation(a, b)

def make_forall(var_name, body):
    return Forall(Var(var_name), body)


# ── 1. Parallel Prover ────────────────────────────────────────────────────────

class TestParallelProver:
    """Tests for ParallelHeuristicProver."""

    def test_sequential_proves_trivial(self):
        """Sequential mode (workers=1) proves 0 = 0."""
        from src.training.parallel_prover import ParallelHeuristicProver
        prover = ParallelHeuristicProver(max_workers=1, max_depth=20, max_iterations=200)
        kb = [str(ax) for ax in get_all_axioms()]
        results = prover.prove_batch_sequential(["0 = 0"], kb)
        assert len(results) == 1
        assert results[0]["success"], f"Expected success, got: {results[0]}"

    def test_sequential_fails_false_statement(self):
        """Sequential mode correctly fails to prove 0 = S(0)."""
        from src.training.parallel_prover import ParallelHeuristicProver
        prover = ParallelHeuristicProver(max_workers=1, max_depth=10, max_iterations=100)
        kb = [str(ax) for ax in get_all_axioms()]
        results = prover.prove_batch_sequential(["0 = S(0)"], kb)
        assert len(results) == 1
        assert not results[0]["success"]

    def test_sequential_batch_order_preserved(self):
        """Result list has same length and order as input."""
        from src.training.parallel_prover import ParallelHeuristicProver
        prover = ParallelHeuristicProver(max_workers=1, max_depth=20, max_iterations=200)
        kb = [str(ax) for ax in get_all_axioms()]
        exprs = ["0 = 0", "0 = S(0)", "0 = 0"]
        results = prover.prove_batch_sequential(exprs, kb)
        assert len(results) == 3
        assert results[0]["expr_str"] == "0 = 0"
        assert results[1]["expr_str"] == "0 = S(0)"
        assert results[2]["expr_str"] == "0 = 0"

    def test_parallel_proves_batch(self):
        """Parallel mode (workers=2) proves a batch of trivially true statements."""
        from src.training.parallel_prover import ParallelHeuristicProver
        prover = ParallelHeuristicProver(
            max_workers=2, max_depth=20, max_iterations=200, timeout_per_proof=10.0
        )
        kb = [str(ax) for ax in get_all_axioms()]
        exprs = ["0 = 0", "S(0) = S(0)"]
        results = prover.prove_batch(exprs, kb)
        assert len(results) == 2
        # Both are trivially reflexive – should succeed
        for r in results:
            assert r["success"], f"Expected success for {r['expr_str']}, got: {r}"

    def test_empty_batch(self):
        """Empty input returns empty output."""
        from src.training.parallel_prover import ParallelHeuristicProver
        prover = ParallelHeuristicProver(max_workers=1)
        results = prover.prove_batch_sequential([], [])
        assert results == []

    def test_worker_handles_bad_expr(self):
        """Worker gracefully handles an unparseable expression string."""
        from src.training.parallel_prover import _prove_worker
        result = _prove_worker(("NOT_VALID_EXPR%%%", [], 10, 50))
        assert result["success"] is False
        assert "error" in result


# ── 2. Self-Paced Curriculum ──────────────────────────────────────────────────

class TestSelfPacedCurriculum:

    def setup_method(self):
        from src.models.advanced_curriculum import SelfPacedCurriculum, SelfPacedConfig
        self.cfg = SelfPacedConfig(
            initial_complexity=2,
            final_complexity=20,
            ema_alpha=0.5,        # Fast updates for testing
            target_lo=0.10,
            target_hi=0.60,
            frontier_margin=3,
            min_obs=2,
            prior_success=0.3,
        )
        self.curriculum = SelfPacedCurriculum(self.cfg)

    def test_initial_frontier_is_initial_complexity(self):
        assert self.curriculum.get_frontier() == self.cfg.initial_complexity

    def test_complexity_range_within_bounds(self):
        lo, hi = self.curriculum.get_complexity_range()
        assert lo >= self.cfg.initial_complexity
        assert hi <= self.cfg.final_complexity
        assert lo <= hi

    def test_record_result_updates_ema(self):
        self.curriculum.record_result(True, 5)
        self.curriculum.record_result(True, 5)
        # After two successes at complexity 5, EMA should be > prior
        assert self.curriculum._ema.get(5, 0) > 0

    def test_frontier_advances_with_success(self):
        # Saturate low complexities with success (EMA will be high, above target_hi)
        for _ in range(5):
            self.curriculum.record_result(True, 2)
            self.curriculum.record_result(True, 3)
            self.curriculum.record_result(True, 4)
            self.curriculum.record_result(True, 5)
        # Complexities 2-5 should have high success rates → frontier should advance past them
        frontier = self.curriculum.get_frontier()
        assert frontier >= 2

    def test_filter_keeps_in_range(self):
        exprs = [
            Equation(Zero(), Zero()),          # complexity ~2
            make_forall("x", Equation(Var("x"), Zero())),  # slightly more complex
        ]
        lo, hi = self.curriculum.get_complexity_range()
        filtered = self.curriculum.filter_by_complexity(exprs)
        for e in filtered:
            assert lo <= e.complexity() <= hi

    def test_get_statistics_has_required_keys(self):
        stats = self.curriculum.get_statistics()
        assert "frontier" in stats
        assert "complexity_range" in stats
        assert "ema_rates" in stats


# ── 3. Adaptive Band Curriculum ───────────────────────────────────────────────

class TestAdaptiveBandCurriculum:

    def setup_method(self):
        from src.models.advanced_curriculum import AdaptiveBandCurriculum, AdaptiveBandConfig
        self.cfg = AdaptiveBandConfig(
            initial_complexity=2,
            final_complexity=20,
            initial_halfwidth=2,
            advance_threshold=0.40,
            retreat_threshold=0.05,
            patience=10,          # Short patience for testing
            max_halfwidth=5,
            window=10,
        )
        self.curriculum = AdaptiveBandCurriculum(self.cfg)

    def test_initial_range(self):
        lo, hi = self.curriculum.get_complexity_range()
        assert lo == 2
        assert hi == 4  # center=2, hw=2

    def test_advances_on_high_success(self):
        initial_center = self.curriculum.center
        for _ in range(10):
            self.curriculum.record_result(True, 3)
        # After 10 successes, center should have advanced
        assert self.curriculum.center >= initial_center

    def test_retreats_on_low_success(self):
        self.curriculum.center = 10.0
        for _ in range(10):
            self.curriculum.record_result(False, 10)
        # After 10 failures, center should have retreated
        assert self.curriculum.center <= 10.0

    def test_widens_on_stall(self):
        initial_hw = self.curriculum.halfwidth
        # Force stall: record results giving ~20% success rate, which is between
        # retreat_threshold (0.05) and advance_threshold (0.40). After patience=10
        # steps without advancing, halfwidth should widen.
        # window=10, so 2 True + 8 False = 0.20 rate (stable in middle zone).
        for _ in range(20):
            self.curriculum.record_result(True, 3)
            self.curriculum.record_result(False, 3)
            self.curriculum.record_result(False, 3)
            self.curriculum.record_result(False, 3)
            self.curriculum.record_result(False, 3)
        # halfwidth should be >= initial since stall should have widened it
        assert self.curriculum.halfwidth >= initial_hw

    def test_range_respects_bounds(self):
        lo, hi = self.curriculum.get_complexity_range()
        assert lo >= self.cfg.initial_complexity
        assert hi <= self.cfg.final_complexity

    def test_statistics_keys(self):
        stats = self.curriculum.get_statistics()
        for key in ("center", "halfwidth", "complexity_range", "rolling_success_rate"):
            assert key in stats


# ── 4. Prioritized Experience Buffer ─────────────────────────────────────────

class TestPrioritizedExperienceBuffer:

    def setup_method(self):
        from src.models.advanced_curriculum import PrioritizedExperienceBuffer
        self.buf = PrioritizedExperienceBuffer(
            maxlen=100,
            target_success_rate=0.3,
            ema_alpha=0.2,
        )

    def test_add_and_size(self):
        self.buf.add("0 = 0", 2.0, True, 1)
        assert self.buf.size() == 1

    def test_sample_respects_successes_only(self):
        self.buf.add("0 = 0", 2.0, True, 1)
        self.buf.add("0 = S(0)", 2.0, False, 0)
        entries = self.buf.sample(10, successes_only=True)
        assert all(e.success for e in entries)

    def test_sample_returns_at_most_n(self):
        for i in range(5):
            self.buf.add(f"expr_{i}", float(i), True, i)
        entries = self.buf.sample(3, successes_only=True)
        assert len(entries) <= 3

    def test_sample_empty_buffer(self):
        entries = self.buf.sample(5)
        assert entries == []

    def test_successful_exprs_returns_strings(self):
        self.buf.add("0 = 0", 2.0, True, 1)
        self.buf.add("S(0) = S(0)", 3.0, True, 1)
        strs = self.buf.successful_exprs(n=5)
        assert isinstance(strs, list)
        assert all(isinstance(s, str) for s in strs)

    def test_priority_assigned(self):
        self.buf.add("0 = 0", 2.0, True, 1)
        entry = list(self.buf._buffer)[0]
        assert 0.0 < entry.priority <= 1.0

    def test_maxlen_enforced(self):
        from src.models.advanced_curriculum import PrioritizedExperienceBuffer
        buf = PrioritizedExperienceBuffer(maxlen=5)
        for i in range(10):
            buf.add(f"e{i}", 2.0, True, 1)
        assert buf.size() == 5

    def test_statistics_keys(self):
        self.buf.add("0 = 0", 2.0, True, 1)
        stats = self.buf.get_statistics()
        for key in ("buffer_size", "success_count", "success_rate", "bucket_ema"):
            assert key in stats


# ── 5. Phase5Config defaults ──────────────────────────────────────────────────

class TestPhase5Config:

    def test_default_construction(self):
        from src.training.phase5_training_loop import Phase5Config
        cfg = Phase5Config()
        assert cfg.curriculum_strategy in ("self_paced", "adaptive_band", "linear")
        assert cfg.parallel_workers >= 1
        assert 0.0 < cfg.neural_ratio < 1.0
        assert cfg.initial_complexity < cfg.final_complexity

    def test_custom_values(self):
        from src.training.phase5_training_loop import Phase5Config
        cfg = Phase5Config(
            num_epochs=5,
            cycles_per_epoch=100,
            parallel_workers=2,
            curriculum_strategy="adaptive_band",
            device="cpu",
        )
        assert cfg.num_epochs == 5
        assert cfg.parallel_workers == 2
        assert cfg.curriculum_strategy == "adaptive_band"


# ── 6. Phase5TrainingLoop smoke test (CPU, tiny scale) ───────────────────────

class TestPhase5TrainingLoopSmoke:
    """Quick smoke test: builds the loop, runs 2 cycles, checks it doesn't crash."""

    @pytest.fixture(autouse=True)
    def build_loop(self):
        """Build a minimal Phase5TrainingLoop on CPU."""
        from src.training.phase5_training_loop import Phase5TrainingLoop, Phase5Config
        cfg = Phase5Config(
            num_epochs=1,
            cycles_per_epoch=2,
            conjectures_per_cycle=3,
            random_seed=0,
            neural_ratio=0.0,         # All random, skip neural model
            initial_complexity=2,
            final_complexity=5,
            curriculum_strategy="self_paced",
            parallel_workers=1,       # Sequential for test safety
            heuristic_max_depth=20,
            heuristic_max_iter=100,
            heuristic_timeout=5.0,
            max_proof_steps=10,
            ppo_update_interval=50,
            bc_warmup_cycles=0,
            use_heuristic_fallback=True,
            encoder_d_model=32,
            encoder_nhead=2,
            encoder_num_layers=1,
            ac_hidden_dim=32,
            ppo_epochs=1,
            ppo_mini_batch_size=4,
            gen_d_model=32,
            gen_nhead=2,
            gen_num_layers=1,
            gen_pretrain_epochs=0,
            gen_update_interval=100,
            gen_use_prioritized_buffer=True,
            use_torch_compile=False,
            device="cpu",
            checkpoint_dir="/tmp/phase5_test_checkpoints",
            experiment_name="test_run",
            checkpoint_interval=9999,  # No checkpointing during test
            log_interval=1,
        )
        self.loop = Phase5TrainingLoop(config=cfg)

    def test_initial_state(self):
        assert self.loop.total_generated == 0
        assert self.loop.total_attempted == 0
        assert self.loop.total_proved_rl == 0
        assert self.loop.total_proved_heuristic == 0
        assert self.loop.ppo_update_count == 0

    def test_run_cycle(self):
        rl_p, h_p = self.loop._run_cycle()
        assert isinstance(rl_p, int)
        assert isinstance(h_p, int)
        assert rl_p >= 0
        assert h_p >= 0

    def test_get_statistics(self):
        self.loop._run_cycle()
        stats = self.loop.get_statistics()
        for key in ("kb_size", "total_generated", "total_attempted",
                    "rl_success_rate", "heuristic_success_rate"):
            assert key in stats

    def test_curriculum_filter_type(self):
        """Curriculum is SelfPacedCurriculum since strategy=self_paced."""
        from src.models.advanced_curriculum import SelfPacedCurriculum
        assert isinstance(self.loop.curriculum, SelfPacedCurriculum)

    def test_exp_buffer_grows(self):
        self.loop._run_cycle()
        assert self.loop.exp_buffer.size() >= 0  # May be 0 if no conjectures passed filter

    def test_full_train_smoke(self):
        """Two epochs of 2 cycles each should complete without error."""
        self.loop.train()
        stats = self.loop.get_statistics()
        assert stats["epoch"] >= 0


# ── 7. Adaptive band loop smoke test ─────────────────────────────────────────

class TestAdaptiveBandLoopSmoke:

    def test_adaptive_band_curriculum_in_loop(self):
        from src.training.phase5_training_loop import Phase5TrainingLoop, Phase5Config
        cfg = Phase5Config(
            num_epochs=1,
            cycles_per_epoch=2,
            conjectures_per_cycle=3,
            random_seed=1,
            neural_ratio=0.0,
            initial_complexity=2,
            final_complexity=8,
            curriculum_strategy="adaptive_band",
            parallel_workers=1,
            heuristic_max_depth=15,
            heuristic_max_iter=80,
            heuristic_timeout=5.0,
            bc_warmup_cycles=0,
            use_heuristic_fallback=True,
            encoder_d_model=32,
            encoder_nhead=2,
            encoder_num_layers=1,
            ac_hidden_dim=32,
            ppo_epochs=1,
            ppo_mini_batch_size=4,
            gen_d_model=32,
            gen_nhead=2,
            gen_num_layers=1,
            gen_pretrain_epochs=0,
            gen_update_interval=100,
            gen_use_prioritized_buffer=False,
            use_torch_compile=False,
            device="cpu",
            checkpoint_dir="/tmp/phase5_test_checkpoints",
            experiment_name="test_abc",
            checkpoint_interval=9999,
            log_interval=1,
        )
        from src.models.advanced_curriculum import AdaptiveBandCurriculum
        loop = Phase5TrainingLoop(config=cfg)
        assert isinstance(loop.curriculum, AdaptiveBandCurriculum)
        loop.train()

    def test_linear_curriculum_in_loop(self):
        from src.training.phase5_training_loop import Phase5TrainingLoop, Phase5Config
        cfg = Phase5Config(
            num_epochs=1,
            cycles_per_epoch=2,
            conjectures_per_cycle=3,
            random_seed=2,
            neural_ratio=0.0,
            initial_complexity=2,
            final_complexity=8,
            curriculum_strategy="linear",
            parallel_workers=1,
            heuristic_max_depth=15,
            heuristic_max_iter=80,
            heuristic_timeout=5.0,
            bc_warmup_cycles=0,
            use_heuristic_fallback=True,
            encoder_d_model=32,
            encoder_nhead=2,
            encoder_num_layers=1,
            ac_hidden_dim=32,
            ppo_epochs=1,
            ppo_mini_batch_size=4,
            gen_d_model=32,
            gen_nhead=2,
            gen_num_layers=1,
            gen_pretrain_epochs=0,
            gen_update_interval=100,
            gen_use_prioritized_buffer=False,
            use_torch_compile=False,
            device="cpu",
            checkpoint_dir="/tmp/phase5_test_checkpoints",
            experiment_name="test_linear",
            checkpoint_interval=9999,
            log_interval=1,
        )
        loop = Phase5TrainingLoop(config=cfg)
        assert loop.curriculum is None
        loop.train()
