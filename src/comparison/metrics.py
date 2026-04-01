"""
Unified comparison metrics for evaluating conjecture generation systems.

All six metrics are computed from a KnowledgeBase snapshot so that any
baseline — random, heuristic, supervised, STP, or full AutoConjecture —
can be measured on an identical footing.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ..logic.expressions import Expression
from ..knowledge.knowledge_base import KnowledgeBase
from ..prover.proof_engine import ProofEngine, ProofResult
from ..prover.tactics import DEFAULT_TACTICS


# ── Triviality tactic list ─────────────────────────────────────────────────────
# Only reflexivity and assumption — anything needing rewrite / simplify / induction
# counts as non-trivial.  We import by name so we don't couple to tactic internals.
def _trivial_tactics():
    return [t for t in DEFAULT_TACTICS if t.name() in ("reflexivity", "assumption")]


@dataclass
class ComparisonSnapshot:
    """Point-in-time measurement for one system at one wall-clock checkpoint."""

    system_name: str
    wall_clock_seconds: float

    # ── Core counts ───────────────────────────────────────────────────────────
    theorems_unique: int            # KB size at snapshot time
    attempted: int                  # Total proof attempts so far
    proved: int                     # Total successful proofs so far

    # ── Derived rates ─────────────────────────────────────────────────────────
    theorems_per_hour: float        # unique theorems / (wall_clock / 3600)
    proof_success_rate: float       # proved / max(attempted, 1)

    # ── Quality metrics ───────────────────────────────────────────────────────
    complexity_histogram: Dict[str, int]   # bucket label -> count
    avg_complexity: float
    non_triviality_rate: float      # fraction not provable by trivial tactics
    diversity_score: float          # Shannon entropy over structural patterns
    provability_at_k: Dict[int, float]     # k -> fraction proved in <=k steps

    def to_dict(self) -> dict:
        return {
            "system": self.system_name,
            "wall_clock_s": round(self.wall_clock_seconds, 2),
            "theorems": self.theorems_unique,
            "attempted": self.attempted,
            "proved": self.proved,
            "theorems_per_hour": round(self.theorems_per_hour, 2),
            "success_rate": round(self.proof_success_rate, 4),
            "avg_complexity": round(self.avg_complexity, 2),
            "non_triviality_rate": round(self.non_triviality_rate, 4),
            "diversity_score": round(self.diversity_score, 4),
            "provability_at_k": {str(k): round(v, 4) for k, v in self.provability_at_k.items()},
            "complexity_histogram": self.complexity_histogram,
        }

    def summary_line(self) -> str:
        p_at_10 = self.provability_at_k.get(10, 0.0)
        p_at_20 = self.provability_at_k.get(20, 0.0)
        return (
            f"{self.system_name:<18} | "
            f"thm={self.theorems_unique:5d} | "
            f"thm/hr={self.theorems_per_hour:7.1f} | "
            f"succ={self.proof_success_rate:6.2%} | "
            f"non-triv={self.non_triviality_rate:6.2%} | "
            f"div={self.diversity_score:.3f} | "
            f"P@10={p_at_10:.2%} | "
            f"P@20={p_at_20:.2%} | "
            f"t={self.wall_clock_seconds:.0f}s"
        )


class ComparisonMetrics:
    """
    Computes all comparison metrics from a running KnowledgeBase.

    Designed to be called at fixed wall-clock intervals by any baseline runner.
    Non-triviality is cached per theorem to keep snapshots cheap after the
    first evaluation.
    """

    # Complexity bucket edges (right-exclusive)
    COMPLEXITY_BUCKETS = [
        (0, 10, "<10"),
        (10, 15, "10-14"),
        (15, 20, "15-19"),
        (20, 25, "20-24"),
        (25, 999, "25+"),
    ]

    PROVABILITY_K_VALUES = [5, 10, 20, 50]

    def __init__(
        self,
        system_name: str,
        trivial_max_depth: int = 5,
        trivial_max_iter: int = 20,
    ):
        self.system_name = system_name
        self.trivial_max_depth = trivial_max_depth
        self.trivial_max_iter = trivial_max_iter

        # Cache: statement_str -> is_non_trivial (bool)
        self._triviality_cache: Dict[str, bool] = {}

    def snapshot(
        self,
        kb: KnowledgeBase,
        wall_clock_seconds: float,
        attempted: int,
        proved: int,
    ) -> ComparisonSnapshot:
        """Compute all metrics and return a snapshot."""
        theorems = kb.get_all_theorems()
        n = len(theorems)

        # Rates
        thm_per_hour = n / max(wall_clock_seconds / 3600.0, 1e-9)
        success_rate = proved / max(attempted, 1)

        # Complexity
        hist = self._complexity_histogram(theorems)
        avg_c = (
            sum(t.complexity for t in theorems) / n if n > 0 else 0.0
        )

        # Non-triviality (uses cache)
        non_triv = self._non_triviality_rate(theorems, kb)

        # Diversity (Shannon entropy on structural patterns)
        diversity = self._diversity_score(theorems)

        # Provability@k (from stored proof lengths)
        p_at_k = self._provability_at_k(theorems)

        return ComparisonSnapshot(
            system_name=self.system_name,
            wall_clock_seconds=wall_clock_seconds,
            theorems_unique=n,
            attempted=attempted,
            proved=proved,
            theorems_per_hour=thm_per_hour,
            proof_success_rate=success_rate,
            complexity_histogram=hist,
            avg_complexity=avg_c,
            non_triviality_rate=non_triv,
            diversity_score=diversity,
            provability_at_k=p_at_k,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _complexity_histogram(self, theorems) -> Dict[str, int]:
        hist = {label: 0 for _, _, label in self.COMPLEXITY_BUCKETS}
        for thm in theorems:
            c = thm.complexity
            for lo, hi, label in self.COMPLEXITY_BUCKETS:
                if lo <= c < hi:
                    hist[label] += 1
                    break
        return hist

    def _non_triviality_rate(self, theorems, kb: KnowledgeBase) -> float:
        """
        A theorem is non-trivial if it cannot be closed by reflexivity/assumption
        alone within a tiny budget.  Results are cached so repeated snapshots
        only check newly added theorems.
        """
        if not theorems:
            return 0.0

        trivial_engine = ProofEngine(
            tactics=_trivial_tactics(),
            max_depth=self.trivial_max_depth,
            max_iterations=self.trivial_max_iter,
            knowledge_base=[],  # No KB — pure structural check
        )

        non_trivial_count = 0
        for thm in theorems:
            key = str(thm.statement)
            if key not in self._triviality_cache:
                proof = trivial_engine.prove(thm.statement)
                is_non_trivial = (proof.result != ProofResult.SUCCESS)
                self._triviality_cache[key] = is_non_trivial
            if self._triviality_cache[key]:
                non_trivial_count += 1

        return non_trivial_count / len(theorems)

    def _diversity_score(self, theorems) -> float:
        """
        Shannon entropy over structural patterns.  Higher = more diverse.
        Patterns replace variables with X and numerals with N (mirrors DiversityFilter).
        """
        if not theorems:
            return 0.0

        pattern_counts: Dict[str, int] = {}
        for thm in theorems:
            p = self._get_pattern(thm.statement)
            pattern_counts[p] = pattern_counts.get(p, 0) + 1

        total = len(theorems)
        entropy = 0.0
        for count in pattern_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalise by log2(total) so score is in [0, 1]
        max_entropy = math.log2(total) if total > 1 else 1.0
        return entropy / max_entropy

    def _get_pattern(self, expr: Expression) -> str:
        s = str(expr)
        for var_name in expr.free_vars():
            s = s.replace(var_name, "X")
        s = s.replace("0", "N")
        # Collapse repeated successor numerals
        import re
        s = re.sub(r"S\(N\)", "N", s)
        return s

    def _provability_at_k(self, theorems) -> Dict[int, float]:
        """
        Fraction of theorems whose stored proof length is <= k.
        Uses the proof_length stored in the KB (reflects actual search effort).
        """
        if not theorems:
            return {k: 0.0 for k in self.PROVABILITY_K_VALUES}

        lengths = [thm.proof.length() for thm in theorems]
        n = len(lengths)
        return {
            k: sum(1 for l in lengths if l <= k) / n
            for k in self.PROVABILITY_K_VALUES
        }
