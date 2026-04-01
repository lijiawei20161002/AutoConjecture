"""
Baseline 3: Neural generator (supervised training on KB) + heuristic prover.

No PPO, no RLProofEngine, no ActorCritic.  Analogous to LeanConjecturer's
generate-then-check approach.  Isolates the contribution of RL from the
neural generator.
"""
from __future__ import annotations

import time
from typing import List

import torch

from ..logic.axioms import get_all_axioms
from ..knowledge.knowledge_base import KnowledgeBase
from ..prover.proof_engine import ProofEngine, ProofResult
from ..training.parallel_prover import ParallelHeuristicProver
from ..generation.neural_generator import NeuralConjectureGenerator
from ..generation.random_generator import RandomConjectureGenerator
from ..generation.heuristics import ComplexityEstimator, DiversityFilter
from ..generation.novelty import NoveltyScorer
from ..models.transformer_generator import TransformerGenerator
from ..models.tokenizer import ExpressionTokenizer
from ..models.generator_trainer import GeneratorTrainer, GeneratorTrainingConfig
from ..comparison.metrics import ComparisonMetrics, ComparisonSnapshot
from .base_runner import BaselineRunner


class SupervisedBaselineRunner(BaselineRunner):
    """
    Neural generator with supervised-only training + heuristic BFS prover.
    No RL.  Analogous to LeanConjecturer's approach.
    """

    @property
    def name(self) -> str:
        return "supervised"

    def setup(self, config: dict) -> None:
        self.device = config.get("device", "cpu")
        var_names = config.get("var_names", ["x", "y", "z", "w"])

        self.kb = KnowledgeBase(axioms=get_all_axioms())

        self.tokenizer = ExpressionTokenizer(max_length=128, var_names=var_names)

        self.gen_model = TransformerGenerator(
            vocab_size=self.tokenizer.vocab_size,
            d_model=config.get("gen_d_model", 256),
            nhead=config.get("gen_nhead", 8),
            num_layers=config.get("gen_num_layers", 6),
        ).to(self.device)

        self.neural_gen = NeuralConjectureGenerator(
            model=self.gen_model,
            tokenizer=self.tokenizer,
            device=self.device,
            temperature=config.get("temperature", 1.0),
            top_k=config.get("top_k", 50),
        )

        self.random_gen = RandomConjectureGenerator(
            min_complexity=config.get("min_complexity", 6),
            max_complexity=config.get("max_complexity", 20),
            var_names=var_names,
            seed=config.get("seed", 42),
        )

        gen_trainer_cfg = GeneratorTrainingConfig(
            learning_rate=config.get("gen_lr", 1e-4),
            batch_size=config.get("gen_batch_size", 32),
            warmup_steps=config.get("gen_warmup_steps", 500),
            device=self.device,
        )
        self.gen_trainer = GeneratorTrainer(
            model=self.gen_model,
            tokenizer=self.tokenizer,
            config=gen_trainer_cfg,
            knowledge_base=self.kb,
        )

        n_workers = config.get("parallel_workers", 2)
        self.parallel_prover = ParallelHeuristicProver(
            max_workers=n_workers,
            max_depth=config.get("max_depth", 50),
            max_iterations=config.get("max_iterations", 500),
            timeout_per_proof=config.get("timeout_per_proof", 30.0),
        )
        self.heuristic_engine = ProofEngine(
            max_depth=config.get("max_depth", 50),
            max_iterations=config.get("max_iterations", 500),
            knowledge_base=self.kb.get_all_statements(),
        )

        self.complexity_est = ComplexityEstimator()
        self.diversity_filter = DiversityFilter(max_similar=config.get("max_similar", 20))
        self.novelty_scorer = NoveltyScorer()
        self.metrics = ComparisonMetrics(self.name)

        self._batch_size: int = config.get("batch_size", 10)
        self._neural_ratio: float = config.get("neural_ratio", 0.5)
        self._update_interval: int = config.get("update_interval", 100)
        self._pretrain_epochs: int = config.get("pretrain_epochs", 0)
        self._attempted: int = 0
        self._proved: int = 0
        self._cycles: int = 0

    def run_for(
        self,
        wall_clock_budget_seconds: float,
        snapshot_interval_seconds: float = 60.0,
    ) -> List[ComparisonSnapshot]:
        snapshots: List[ComparisonSnapshot] = []

        # Optional pretrain on any existing KB content
        if self._pretrain_epochs > 0 and self.kb.size() > 0:
            print(f"[{self.name}] Pre-training on {self.kb.size()} existing theorems...")
            self.gen_trainer.train_on_knowledge_base(
                num_epochs=self._pretrain_epochs,
                curriculum_strategy="complexity",
            )

        t_start = time.time()
        t_last_snap = t_start

        while True:
            now = time.time()
            if (now - t_start) >= wall_clock_budget_seconds:
                break

            self._cycles += 1

            # ── generate ──────────────────────────────────────────────────
            n_neural = int(self._batch_size * self._neural_ratio)
            n_random = self._batch_size - n_neural

            self.neural_gen.eval_mode()
            conjectures = []
            if n_neural > 0:
                conjectures.extend(c for c in self.neural_gen.generate(n_neural) if c is not None)
            if n_random > 0:
                conjectures.extend(self.random_gen.generate(n_random))

            # ── filter ────────────────────────────────────────────────────
            filtered = []
            for c in conjectures:
                if c is None:
                    continue
                if not self.complexity_est.is_well_formed(c):
                    continue
                if self.novelty_scorer.score(c) < 0.3:
                    continue
                if not self.diversity_filter.should_keep(c):
                    continue
                if self.kb.contains(c):
                    continue
                filtered.append(c)

            if not filtered:
                continue

            # ── prove (parallel heuristic only — NO RL) ───────────────────
            kb_stmts = self.kb.get_all_statements()
            kb_strs = [str(s) for s in kb_stmts]
            expr_strs = [str(c) for c in filtered]
            self._attempted += len(filtered)

            try:
                results = self.parallel_prover.prove_batch(expr_strs, kb_strs)
            except Exception:
                results = self.parallel_prover.prove_batch_sequential(expr_strs, kb_strs)

            for c, res in zip(filtered, results):
                if res["success"] and not self.kb.contains(c):
                    complexity = self.complexity_est.estimate(c)
                    from ..prover.proof_engine import Proof, ProofResult as PR
                    stub = Proof(statement=c, steps=[], result=PR.SUCCESS)
                    added = self.kb.add_theorem(c, stub, complexity, 0, self._attempted)
                    if added:
                        self._proved += 1
                        self.novelty_scorer.add(c)

            # ── supervised generator update (NO RL/PPO) ───────────────────
            if self._cycles % self._update_interval == 0:
                self._supervised_update()

            # ── snapshot ──────────────────────────────────────────────────
            now = time.time()
            if (now - t_last_snap) >= snapshot_interval_seconds:
                snap = self.metrics.snapshot(
                    self.kb, now - t_start, self._attempted, self._proved
                )
                snapshots.append(snap)
                print(snap.summary_line(), flush=True)
                t_last_snap = now

        final_snap = self.metrics.snapshot(
            self.kb, time.time() - t_start, self._attempted, self._proved
        )
        snapshots.append(final_snap)
        return snapshots

    def _supervised_update(self):
        """Pure cross-entropy update on recent KB theorems.  No policy gradient."""
        recent_count = min(50, self.kb.size())
        if recent_count < 5:
            return
        recent = self.kb.get_all_theorems()[-recent_count:]
        exprs = [t.statement for t in recent]
        self.neural_gen.train_mode()
        try:
            loss = self.gen_trainer._train_batch(exprs)
            print(f"  [{self.name}] supervised update loss={loss:.4f}", flush=True)
        except Exception as e:
            print(f"  [{self.name}] supervised update failed: {e}", flush=True)
        finally:
            self.neural_gen.eval_mode()

    def get_final_kb(self) -> KnowledgeBase:
        return self.kb
