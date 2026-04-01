"""
Lean 4-aware conjecture generator.

Three strategies:
  1. translate  — Convert AutoConjecture Peano AST → Lean 4 via PeanoToLean4Translator
  2. template   — Lean 4 string templates for standard Mathlib theorem patterns
  3. combined   — Mix of (1) and (2)
"""
from __future__ import annotations

import random
from typing import List, Optional

from ..generation.random_generator import RandomConjectureGenerator
from ..generation.algebraic_templates import AlgebraicTemplateGenerator
from .ast_translator import PeanoToLean4Translator, TranslationError


# ── Lean 4 string templates ────────────────────────────────────────────────────
# These are complete theorem declarations (without the `by <tac>` suffix).
# The prover will append tactics.

LEAN4_TEMPLATES = [
    # Commutativity
    "theorem add_comm_var (x y : Nat) : x + y = y + x",
    "theorem mul_comm_var (x y : Nat) : x * y = y * x",
    # Associativity
    "theorem add_assoc_var (x y z : Nat) : (x + y) + z = x + (y + z)",
    "theorem mul_assoc_var (x y z : Nat) : (x * y) * z = x * (y * z)",
    # Identity
    "theorem add_zero_var (x : Nat) : x + 0 = x",
    "theorem zero_add_var (x : Nat) : 0 + x = x",
    "theorem mul_one_var (x : Nat) : x * 1 = x",
    "theorem one_mul_var (x : Nat) : 1 * x = x",
    # Absorption
    "theorem mul_zero_var (x : Nat) : x * 0 = 0",
    "theorem zero_mul_var (x : Nat) : 0 * x = 0",
    # Distributivity
    "theorem left_distrib_var (x y z : Nat) : x * (y + z) = x * y + x * z",
    "theorem right_distrib_var (x y z : Nat) : (x + y) * z = x * z + y * z",
    # Successor
    "theorem succ_add_var (x y : Nat) : Nat.succ x + y = Nat.succ (x + y)",
    "theorem add_succ_var (x y : Nat) : x + Nat.succ y = Nat.succ (x + y)",
    "theorem succ_mul_var (x y : Nat) : Nat.succ x * y = x * y + y",
    "theorem mul_succ_var (x y : Nat) : x * Nat.succ y = x * y + x",
    # Double
    "theorem double_var (x : Nat) : x + x = 2 * x",
    "theorem double_var2 (x : Nat) : 2 * x = x + x",
    # Mixed
    "theorem add_mul_var (x y z : Nat) : x * y + x * z = x * (y + z)",
    "theorem add_comm_assoc_var (x y z : Nat) : x + y + z = y + x + z",
    "theorem mul_add_var (x y : Nat) : x * (x + y) = x * x + x * y",
    # Nested commutativity
    "theorem add_right_comm_var (x y z : Nat) : x + y + z = x + z + y",
    "theorem mul_right_comm_var (x y z : Nat) : x * y * z = x * z * y",
    # Square-like
    "theorem sq_add_var (x y : Nat) : (x + y) * (x + y) = x * x + 2 * x * y + y * y",
    # Cancellation-style
    "theorem add_left_cancel_var (x y z : Nat) : x + y = x + z → y = z",
    "theorem add_right_cancel_var (x y z : Nat) : y + x = z + x → y = z",
    # Transitivity-style
    "theorem add_le_add_var (x y z : Nat) : x ≤ y → x + z ≤ y + z",
]


class Lean4ConjectureGenerator:
    """
    Generates conjecture strings that are syntactically valid Lean 4.

    Combines translated Peano conjectures and Lean 4 string templates.
    """

    def __init__(
        self,
        var_names: Optional[List[str]] = None,
        seed: int = 42,
        translate_ratio: float = 0.6,  # Fraction from translation vs templates
    ):
        self.var_names = var_names or ["x", "y", "z", "w"]
        self.seed = seed
        self.translate_ratio = translate_ratio
        random.seed(seed)

        self.translator = PeanoToLean4Translator(default_tactic="auto")

        # Peano generators for translation
        self.random_gen = RandomConjectureGenerator(
            min_complexity=6, max_complexity=20, var_names=self.var_names, seed=seed
        )
        self.template_gen = AlgebraicTemplateGenerator(
            var_names=self.var_names, seed=seed
        )

        # Pre-expand Lean 4 templates into full theorem strings (without tactic)
        self._lean4_templates = list(LEAN4_TEMPLATES)
        self._template_cursor = 0

    def generate(self, n: int) -> List[str]:
        """
        Generate n Lean 4 theorem strings (without `:= by <tac>` suffix).
        Returns only syntactically valid translations.
        """
        n_from_peano = int(n * self.translate_ratio)
        n_from_templates = n - n_from_peano

        results: List[str] = []

        if n_from_peano > 0:
            results.extend(self._generate_from_peano(n_from_peano))
        if n_from_templates > 0:
            results.extend(self._generate_from_templates(n_from_templates))

        # If we came up short (translation failures), pad from templates
        shortfall = n - len(results)
        if shortfall > 0:
            results.extend(self._generate_from_templates(shortfall))

        return results[:n]

    def generate_with_multi_tactic(self, n: int) -> List[str]:
        """
        Like generate() but each theorem string includes a multi-tactic
        `first | tac1 | tac2 | ...` proof block.
        """
        theorems = []
        exprs_to_translate = []

        # Collect Peano ASTs
        peano = self.random_gen.generate(n)
        for expr in peano:
            if expr is None:
                continue
            try:
                thm = self.translator.to_theorem_multi_tactic(expr)
                theorems.append(thm)
            except TranslationError:
                continue

        # Fill with templates
        tpl_needed = n - len(theorems)
        if tpl_needed > 0:
            for tstr in self._generate_from_templates(tpl_needed):
                from .lean4_prover import DEFAULT_TACTIC_PORTFOLIO
                tactics = "; try ".join(DEFAULT_TACTIC_PORTFOLIO[:4])
                theorems.append(f"{tstr} := by first | (ring) | (omega) | (simp) | (decide)")

        return theorems[:n]

    def all_template_theorems(self) -> List[str]:
        """Return all hard-coded Lean 4 template theorem strings."""
        return list(self._lean4_templates)

    # ── private ────────────────────────────────────────────────────────────────

    def _generate_from_peano(self, n: int) -> List[str]:
        """Translate Peano AST conjectures to Lean 4 strings."""
        # Mix random and algebraic-template Peano conjectures
        n_rand = n // 2
        n_tpl = n - n_rand
        conjectures = (
            self.random_gen.generate(n_rand)
            + self.template_gen.generate(n_tpl)
        )
        results: List[str] = []
        counter = 0
        for expr in conjectures:
            if expr is None:
                continue
            try:
                thm = self.translator.to_theorem(
                    expr, name=f"auto_thm_{counter}", tactic=None
                )
                # Strip the `:= by <tac>` suffix — prover appends its own
                if " := " in thm:
                    thm_no_proof = thm.split(" := ")[0].strip()
                else:
                    thm_no_proof = thm
                results.append(thm_no_proof)
                counter += 1
            except TranslationError:
                continue
        return results

    def _generate_from_templates(self, n: int) -> List[str]:
        """Cycle through hard-coded Lean 4 templates."""
        results: List[str] = []
        total = len(self._lean4_templates)
        if total == 0:
            return results
        for i in range(n):
            results.append(self._lean4_templates[self._template_cursor % total])
            self._template_cursor += 1
        return results
