"""
Baseline 1: Random conjecture generation + heuristic BFS prover.

Corresponds to AutoConjecture Phase 1.  No training, no neural network,
no curriculum.  Acts as the lower-bound reference point.
"""
from __future__ import annotations

import time
from typing import List

from ..logic.axioms import get_all_axioms
from ..knowledge.knowledge_base import KnowledgeBase
from ..prover.proof_engine import ProofEngine, ProofResult
from ..generation.random_generator import RandomConjectureGenerator
from ..generation.heuristics import ComplexityEstimator, DiversityFilter
from ..generation.novelty import NoveltyScorer
from ..comparison.metrics import ComparisonMetrics, ComparisonSnapshot
from .base_runner import BaselineRunner


class RandomBaselineRunner(BaselineRunner):
    """
    Random generation + heuristic BFS proving.
    No learning of any kind.  Serves as the floor baseline.
    """

    @property
    def name(self) -> str:
        return "random"

    def setup(self, config: dict) -> None:
        self.kb = KnowledgeBase(axioms=get_all_axioms())

        self.generator = RandomConjectureGenerator(
            min_complexity=config.get("min_complexity", 6),
            max_complexity=config.get("max_complexity", 20),
            var_names=config.get("var_names", ["x", "y", "z", "w"]),
            seed=config.get("seed", 42),
        )

        self.engine = ProofEngine(
            max_depth=config.get("max_depth", 50),
            max_iterations=config.get("max_iterations", 500),
            knowledge_base=self.kb.get_all_statements(),
        )

        self.complexity_est = ComplexityEstimator()
        self.diversity_filter = DiversityFilter(max_similar=config.get("max_similar", 20))
        self.novelty_scorer = NoveltyScorer()
        self.metrics = ComparisonMetrics(self.name)

        self._batch_size: int = config.get("batch_size", 10)
        self._attempted: int = 0
        self._proved: int = 0

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
            elapsed = now - t_start
            if elapsed >= wall_clock_budget_seconds:
                break

            # ── generate ──────────────────────────────────────────────────
            conjectures = self.generator.generate(self._batch_size)

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

        # Final snapshot
        final_snap = self.metrics.snapshot(
            self.kb, time.time() - t_start, self._attempted, self._proved
        )
        snapshots.append(final_snap)
        return snapshots

    def get_final_kb(self) -> KnowledgeBase:
        return self.kb
