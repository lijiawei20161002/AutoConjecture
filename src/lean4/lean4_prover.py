"""
Lean 4 tactic prover: attempts to prove a theorem statement using a
fixed portfolio of tactics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from .repl_interface import Lean4REPLInterface, Lean4Result, Lean4NotAvailable


@dataclass
class Lean4ProofResult:
    """Result of one theorem proof attempt."""
    success: bool
    tactic_used: Optional[str]       # First tactic that worked
    proof_str: Optional[str]         # Full Lean 4 theorem string with proof
    num_tactics_tried: int
    time_taken_seconds: float
    error_messages: List[str]        # Errors from failed tactic attempts


# Default tactic portfolio, ordered by likelihood of success on Peano arithmetic
DEFAULT_TACTIC_PORTFOLIO = [
    "decide",
    "norm_num",
    "ring",
    "omega",
    "simp",
    "simp [Nat.add_comm, Nat.add_assoc, Nat.mul_comm, Nat.mul_assoc]",
    "simp [Nat.add_comm]",
    "simp [Nat.mul_comm]",
    "ring_nf; simp",
    "induction x <;> simp [*]",
    "induction x <;> ring",
    "aesop",
]


class Lean4TacticProver:
    """
    Attempts to prove Lean 4 theorem statements using a tactic portfolio.

    For each theorem, tries tactics in priority order until one succeeds
    or all fail.  Parallelising across tactics is not done here because
    the REPL is shared; instead, use multiple Lean4TacticProver instances
    with separate REPLs for parallel evaluation.
    """

    def __init__(
        self,
        repl: Lean4REPLInterface,
        tactic_portfolio: Optional[List[str]] = None,
        timeout_per_tactic: float = 30.0,
    ):
        self.repl = repl
        self.portfolio = tactic_portfolio or DEFAULT_TACTIC_PORTFOLIO
        self.timeout_per_tactic = timeout_per_tactic

    def prove(self, theorem_str: str) -> Lean4ProofResult:
        """
        Try each tactic in the portfolio until one succeeds.

        Args:
            theorem_str: A Lean 4 theorem declaration WITHOUT the `:= by <tac>` suffix.
                         E.g. "theorem t (x y : Nat) : x + y = y + x"
                         The prover appends each tactic in turn.

        Returns:
            Lean4ProofResult with success=True and tactic_used set if any tactic works.
        """
        t_start = time.time()
        errors: List[str] = []

        # Strip any existing `:= by ...` suffix so we can append our own
        base = theorem_str
        for sep in [" := by", ":= by", ":="]:
            idx = base.find(sep)
            if idx != -1:
                base = base[:idx].rstrip()
                break

        for tactic in self.portfolio:
            full_code = f"{base} := by {tactic}"
            result = self._try_single(full_code)

            if result.success:
                return Lean4ProofResult(
                    success=True,
                    tactic_used=tactic,
                    proof_str=full_code,
                    num_tactics_tried=len(errors) + 1,
                    time_taken_seconds=time.time() - t_start,
                    error_messages=errors,
                )
            if result.error_message:
                errors.append(f"[{tactic}] {result.error_message[:200]}")

        return Lean4ProofResult(
            success=False,
            tactic_used=None,
            proof_str=None,
            num_tactics_tried=len(self.portfolio),
            time_taken_seconds=time.time() - t_start,
            error_messages=errors,
        )

    def prove_with_tactic(self, theorem_str: str, tactic: str) -> Lean4Result:
        """Attempt proof with exactly one tactic (no fallback)."""
        base = theorem_str
        for sep in [" := by", ":= by", ":="]:
            idx = base.find(sep)
            if idx != -1:
                base = base[:idx].rstrip()
                break
        return self._try_single(f"{base} := by {tactic}")

    def prove_batch(self, theorem_strs: List[str]) -> List[Lean4ProofResult]:
        """Prove a list of theorems sequentially (safe with shared REPL)."""
        return [self.prove(t) for t in theorem_strs]

    # ── private ────────────────────────────────────────────────────────────────

    def _try_single(self, full_code: str) -> Lean4Result:
        try:
            return self.repl.check_theorem(full_code)
        except Lean4NotAvailable:
            raise
        except Exception as e:
            from .repl_interface import Lean4Result
            return Lean4Result(
                success=False,
                error_message=str(e),
                proof_term=None,
                time_taken_seconds=0.0,
            )
