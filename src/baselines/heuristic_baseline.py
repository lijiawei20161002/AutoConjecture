"""
Baseline 2: Template-based conjecture generation + heuristic BFS prover.

No neural network, no RL training.  Analogous to TxGraffiti's symbolic
heuristic approach: all conjectures come from a fixed algebraic template
bank that is cycled through repeatedly.
"""
from __future__ import annotations

import time
from typing import List

from ..logic.axioms import get_all_axioms
from ..knowledge.knowledge_base import KnowledgeBase
from ..prover.proof_engine import ProofEngine, ProofResult
from ..generation.algebraic_templates import AlgebraicTemplateGenerator
from ..generation.random_generator import RandomConjectureGenerator
from ..generation.heuristics import ComplexityEstimator, DiversityFilter
from ..generation.novelty import NoveltyScorer
from ..comparison.metrics import ComparisonMetrics, ComparisonSnapshot
from .base_runner import BaselineRunner


class HeuristicBaselineRunner(BaselineRunner):
    """
    Heuristic template generation + heuristic BFS proving.
    No learning.  Analogous to TxGraffiti / The Optimist.
    """

    @property
    def name(self) -> str:
        return "heuristic"

    def setup(self, config: dict) -> None:
        self.kb = KnowledgeBase(axioms=get_all_axioms())

        var_names = config.get("var_names", ["x", "y", "z", "w"])
        self.template_gen = AlgebraicTemplateGenerator(
            var_names=var_names,
            seed=config.get("seed", 42),
        )
        # Random fallback once templates are exhausted (keeps generator alive)
        self.random_gen = RandomConjectureGenerator(
            min_complexity=config.get("min_complexity", 6),
            max_complexity=config.get("max_complexity", 20),
            var_names=var_names,
            seed=config.get("seed", 42),
        )

        self.engine = ProofEngine(
            max_depth=config.get("max_depth", 50),
            max_iterations=config.get("max_iterations", 500),
            knowledge_base=self.kb.get_all_statements(),
        )

        self.complexity_est = ComplexityEstimator()
        self.diversity_filter = DiversityFilter(max_similar=config.get("max_similar", 50))
        self.novelty_scorer = NoveltyScorer()
        self.metrics = ComparisonMetrics(self.name)

        self._batch_size: int = config.get("batch_size", 10)
        # Fraction of each batch drawn from templates (rest from random)
        self._template_ratio: float = config.get("template_ratio", 0.7)
        self._attempted: int = 0
        self._proved: int = 0
        self._cycles: int = 0

        print(
            f"[{self.name}] Template pool: {self.template_gen.pool_size()} conjectures",
            flush=True,
        )

    def run_for(
        self,
        wall_clock_budget_seconds: float,
        snapshot_interval_seconds: float = 60.0,
    ) -> List[ComparisonSnapshot]:
        snapshots: List[ComparisonSnapshot] = []
        t_start = time.time()
        t_last_snap = t_start

        while True:
            now = time.time()
            if (now - t_start) >= wall_clock_budget_seconds:
                break

            self._cycles += 1

            # ── generate ──────────────────────────────────────────────────
            n_template = int(self._batch_size * self._template_ratio)
            n_random = self._batch_size - n_template

            conjectures = []
            if n_template > 0:
                conjectures.extend(self.template_gen.generate(n_template))
            if n_random > 0:
                conjectures.extend(self.random_gen.generate(n_random))

            # ── filter ────────────────────────────────────────────────────
            filtered = []
            for c in conjectures:
                if c is None:
                    continue
                if not self.complexity_est.is_well_formed(c):
                    continue
                if self.novelty_scorer.score(c) < 0.1:
                    continue
                if not self.diversity_filter.should_keep(c):
                    continue
                if self.kb.contains(c):
                    continue
                filtered.append(c)

            # ── prove ─────────────────────────────────────────────────────
            kb_stmts = self.kb.get_all_statements()
            self.engine.knowledge_base = kb_stmts

            for c in filtered:
                self._attempted += 1
                proof = self.engine.prove(c)
                if proof.result == ProofResult.SUCCESS:
                    complexity = self.complexity_est.estimate(c)
                    added = self.kb.add_theorem(c, proof, complexity, 0, self._attempted)
                    if added:
                        self._proved += 1
                        self.novelty_scorer.add(c)

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

    def get_final_kb(self) -> KnowledgeBase:
        return self.kb
