"""
Translator: AutoConjecture Peano AST → Lean 4 Mathlib syntax.

Mapping:
  Zero()          → 0
  Succ(Zero())    → 1  (Succ(Succ(Zero())) → 2, etc.)
  Succ(expr)      → Nat.succ (expr)  when expr is not a numeral
  Var("x")        → x
  Add(a, b)       → a + b
  Mul(a, b)       → a * b
  Equation(l, r)  → l = r
  Forall(x, body) → (x : ℕ) → body   [in theorem signature]
                    ∀ x : ℕ, body     [inside body]
  Exists(x, body) → ∃ x : ℕ, body
  And(a, b)       → a ∧ b
  Or(a, b)        → a ∨ b
  Not(b)          → ¬b
  Implies(h, c)   → h → c

The full theorem wrapper:
  Peano: ∀x.∀y. (x + y = y + x)
  Lean4: theorem auto_thm (x y : Nat) : x + y = y + x := by ring
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from ..logic.terms import Term, Var, Zero, Succ, Add, Mul
from ..logic.expressions import (
    Expression, Equation, Forall, Exists,
    And, Or, Not, Implies,
)


class TranslationError(ValueError):
    """Raised when an expression cannot be translated to Lean 4."""
    pass


class PeanoToLean4Translator:
    """
    Translates AutoConjecture Peano arithmetic AST nodes to Lean 4 strings.

    Example
    -------
    >>> t = PeanoToLean4Translator()
    >>> expr = Forall(Var("x"), Forall(Var("y"), Equation(Add(Var("x"), Var("y")), Add(Var("y"), Var("x")))))
    >>> print(t.to_theorem(expr, name="add_comm"))
    theorem add_comm (x y : Nat) : x + y = y + x := by ring
    """

    TACTIC_PORTFOLIO = ["decide", "ring", "omega", "simp", "norm_num", "aesop"]

    def __init__(self, default_tactic: str = "auto"):
        """
        Args:
            default_tactic: Tactic to put in the `by <tac>` clause.
                "auto" means the translator will infer the best tactic.
        """
        self.default_tactic = default_tactic

    # ── Public API ─────────────────────────────────────────────────────────────

    def to_theorem(
        self,
        expr: Expression,
        name: str = "auto_thm",
        tactic: Optional[str] = None,
    ) -> str:
        """
        Wrap an Expression as a complete Lean 4 theorem declaration.

        All top-level Forall quantifiers become explicit theorem arguments.
        Any remaining free variables become additional arguments.

        Returns a string like:
            theorem add_comm (x y : Nat) : x + y = y + x := by ring
        """
        # Peel off top-level universal quantifiers into explicit args
        bound_vars, body = self._strip_foralls(expr)
        # Collect any remaining free vars in the body (shouldn't exist if
        # the expression is fully quantified, but handle it gracefully)
        free = body.free_vars() - {v for v, _ in bound_vars}
        for fv in sorted(free):
            bound_vars.append((fv, "Nat"))

        # Build signature: (x : Nat) (y : Nat) ...
        if bound_vars:
            args = " ".join(f"({v} : Nat)" for v, _ in bound_vars)
            sig = f"theorem {name} {args} : "
        else:
            sig = f"theorem {name} : "

        # Translate the body
        body_str = self.translate_expression(body)

        # Choose tactic
        tac = tactic or (
            self.default_tactic if self.default_tactic != "auto"
            else self._infer_tactic(body)
        )
        proof = f"by {tac}"

        return f"{sig}{body_str} := {proof}"

    def to_theorem_multi_tactic(
        self,
        expr: Expression,
        name: str = "auto_thm",
    ) -> str:
        """
        Like to_theorem but wraps the proof in a `first | tac1 | tac2 | ...`
        block to try multiple tactics in order.
        """
        bound_vars, body = self._strip_foralls(expr)
        free = body.free_vars() - {v for v, _ in bound_vars}
        for fv in sorted(free):
            bound_vars.append((fv, "Nat"))

        args = " ".join(f"({v} : Nat)" for v, _ in bound_vars)
        sig = f"theorem {name} {args} : " if args else f"theorem {name} : "
        body_str = self.translate_expression(body)

        tactics = self._suggest_tactics(body)
        if len(tactics) == 1:
            proof = f"by {tactics[0]}"
        else:
            inner = " | ".join(f"({t})" for t in tactics)
            proof = f"by first | {inner}"

        return f"{sig}{body_str} := {proof}"

    def translate_expression(self, expr: Expression) -> str:
        """Convert an Expression to a Lean 4 proposition string (no theorem wrapper)."""
        if isinstance(expr, Equation):
            l = self.translate_term(expr.left)
            r = self.translate_term(expr.right)
            return f"{l} = {r}"
        elif isinstance(expr, Forall):
            v = expr.var.name
            body = self.translate_expression(expr.body)
            return f"∀ {v} : Nat, {body}"
        elif isinstance(expr, Exists):
            v = expr.var.name
            body = self.translate_expression(expr.body)
            return f"∃ {v} : Nat, {body}"
        elif isinstance(expr, And):
            l = self.translate_expression(expr.left)
            r = self.translate_expression(expr.right)
            return f"({l}) ∧ ({r})"
        elif isinstance(expr, Or):
            l = self.translate_expression(expr.left)
            r = self.translate_expression(expr.right)
            return f"({l}) ∨ ({r})"
        elif isinstance(expr, Not):
            b = self.translate_expression(expr.body)
            return f"¬({b})"
        elif isinstance(expr, Implies):
            h = self.translate_expression(expr.hypothesis)
            c = self.translate_expression(expr.conclusion)
            return f"({h}) → ({c})"
        else:
            raise TranslationError(f"Unknown expression type: {type(expr).__name__}")

    def translate_term(self, term: Term) -> str:
        """Convert a Term to a Lean 4 expression string."""
        if isinstance(term, Var):
            return term.name
        elif isinstance(term, Zero):
            return "0"
        elif isinstance(term, Succ):
            # Try to fold into a numeral
            n = self._as_numeral(term)
            if n is not None:
                return str(n)
            inner = self.translate_term(term.term)
            return f"Nat.succ ({inner})"
        elif isinstance(term, Add):
            l = self._term_maybe_paren(term.left)
            r = self._term_maybe_paren(term.right)
            return f"{l} + {r}"
        elif isinstance(term, Mul):
            l = self._term_maybe_paren(term.left)
            r = self._term_maybe_paren(term.right)
            return f"{l} * {r}"
        else:
            raise TranslationError(f"Unknown term type: {type(term).__name__}")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _term_maybe_paren(self, term: Term) -> str:
        """Parenthesise compound terms (Add/Mul) to avoid precedence issues."""
        s = self.translate_term(term)
        if isinstance(term, (Add, Mul)):
            return f"({s})"
        return s

    def _as_numeral(self, term: Term) -> Optional[int]:
        """
        If term is a Peano numeral (Succ^n(Zero)), return the integer n.
        Returns None otherwise.
        """
        n = 0
        t = term
        while isinstance(t, Succ):
            n += 1
            t = t.term
        if isinstance(t, Zero):
            return n
        return None

    def _strip_foralls(
        self, expr: Expression
    ) -> Tuple[List[Tuple[str, str]], Expression]:
        """
        Peel off top-level Forall quantifiers.

        Returns:
            bound_vars: list of (var_name, type_str) in order
            body: the innermost non-Forall expression
        """
        bound: List[Tuple[str, str]] = []
        current = expr
        while isinstance(current, Forall):
            bound.append((current.var.name, "Nat"))
            current = current.body
        return bound, current

    def _infer_tactic(self, body: Expression) -> str:
        """
        Heuristically choose a likely-to-succeed tactic.

        Priority order:
          1. decide  — ground decidable propositions
          2. ring    — ring equalities (polynomial identities over Nat)
          3. omega   — linear arithmetic
          4. simp    — general simplification
          5. aesop   — last resort
        """
        tactics = self._suggest_tactics(body)
        return tactics[0] if tactics else "simp"

    def _suggest_tactics(self, body: Expression) -> List[str]:
        """Return an ordered list of tactics likely to succeed for this body."""
        # Check for free variables
        has_vars = bool(body.free_vars())
        # Check expression type
        is_eq = isinstance(body, Equation)
        is_implies = isinstance(body, Implies)
        is_and = isinstance(body, And)
        is_exists = isinstance(body, Exists)

        # Detect operations used
        body_str = str(body)
        has_mul = "*" in body_str or "Mul" in body_str or "mul" in body_str.lower()
        has_add = "+" in body_str or "Add" in body_str or "add" in body_str.lower()
        has_succ = "S(" in body_str or "Succ" in body_str

        tactics = []

        if not has_vars:
            # Ground proposition → decide or norm_num
            tactics.extend(["decide", "norm_num"])

        if is_eq:
            if has_mul:
                # Ring can handle polynomial identities
                tactics.append("ring")
            if has_add and not has_mul:
                # Linear arithmetic
                tactics.append("omega")
            tactics.extend(["simp", "ring", "omega", "norm_num"])
        elif is_implies:
            tactics.extend(["omega", "simp", "tauto", "aesop"])
        elif is_exists:
            tactics.extend(["exact ⟨0, by simp⟩", "simp", "aesop"])
        elif is_and:
            tactics.extend(["constructor <;> simp", "simp", "aesop"])
        else:
            tactics.extend(["simp", "aesop"])

        if "induction" not in str(tactics):
            # Add an induction-based tactic as fallback for universally quantified
            tactics.append("simp [Nat.add_comm, Nat.add_assoc, Nat.mul_comm]")

        # Deduplicate while preserving order
        seen: Set[str] = set()
        ordered: List[str] = []
        for t in tactics:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        return ordered
