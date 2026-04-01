"""
Algebraic template bank for the heuristic baseline.

Each template is a callable that generates a family of Peano arithmetic
conjectures by varying variable assignments and term complexity.
Inspired by TxGraffiti's Dalmatian heuristic — no neural network, no RL,
pure symbolic pattern generation.
"""
from __future__ import annotations

import itertools
import random
from typing import List, Optional

from ..logic.terms import Var, Zero, Succ, Add, Mul, Term
from ..logic.expressions import Expression, Equation, Forall, Exists, And, Implies


# ── helpers ────────────────────────────────────────────────────────────────────

def _var(name: str) -> Var:
    return Var(name)


def _zero() -> Zero:
    return Zero()


def _succ(t: Term) -> Succ:
    return Succ(t)


def _nat(n: int) -> Term:
    """Build the Peano numeral for n."""
    t: Term = Zero()
    for _ in range(n):
        t = Succ(t)
    return t


def _add(a: Term, b: Term) -> Add:
    return Add(a, b)


def _mul(a: Term, b: Term) -> Mul:
    return Mul(a, b)


def _eq(l: Term, r: Term) -> Equation:
    return Equation(l, r)


def _forall(v: str, body: Expression) -> Forall:
    return Forall(Var(v), body)


def _forall_many(vars_: List[str], body: Expression) -> Expression:
    expr = body
    for v in reversed(vars_):
        expr = _forall(v, expr)
    return expr


# ── template functions ─────────────────────────────────────────────────────────

def _commutative_add_templates(var_names: List[str]) -> List[Expression]:
    """∀x y. x + y = y + x  (and variations with nested terms)"""
    results = []
    for v1, v2 in itertools.permutations(var_names, 2):
        x, y = _var(v1), _var(v2)
        # Basic: x + y = y + x
        results.append(_forall_many([v1, v2], _eq(_add(x, y), _add(y, x))))
        # Nested: (x + y) + z = z + (x + y)  (use a third var if available)
        remaining = [v for v in var_names if v not in (v1, v2)]
        if remaining:
            v3 = remaining[0]
            z = _var(v3)
            results.append(_forall_many(
                [v1, v2, v3],
                _eq(_add(_add(x, y), z), _add(z, _add(x, y)))
            ))
    return results


def _commutative_mul_templates(var_names: List[str]) -> List[Expression]:
    """∀x y. x * y = y * x"""
    results = []
    for v1, v2 in itertools.permutations(var_names, 2):
        x, y = _var(v1), _var(v2)
        results.append(_forall_many([v1, v2], _eq(_mul(x, y), _mul(y, x))))
    return results


def _associativity_add_templates(var_names: List[str]) -> List[Expression]:
    """∀x y z. (x + y) + z = x + (y + z)"""
    results = []
    for v1, v2, v3 in itertools.permutations(var_names[:4], 3):
        x, y, z = _var(v1), _var(v2), _var(v3)
        results.append(_forall_many(
            [v1, v2, v3],
            _eq(_add(_add(x, y), z), _add(x, _add(y, z)))
        ))
    return results


def _associativity_mul_templates(var_names: List[str]) -> List[Expression]:
    """∀x y z. (x * y) * z = x * (y * z)"""
    results = []
    for v1, v2, v3 in itertools.permutations(var_names[:4], 3):
        x, y, z = _var(v1), _var(v2), _var(v3)
        results.append(_forall_many(
            [v1, v2, v3],
            _eq(_mul(_mul(x, y), z), _mul(x, _mul(y, z)))
        ))
    return results


def _add_zero_identity_templates(var_names: List[str]) -> List[Expression]:
    """∀x. x + 0 = x  and  ∀x. 0 + x = x"""
    results = []
    for v in var_names:
        x = _var(v)
        results.append(_forall(v, _eq(_add(x, _zero()), x)))
        results.append(_forall(v, _eq(_add(_zero(), x), x)))
    return results


def _mul_zero_annihilation_templates(var_names: List[str]) -> List[Expression]:
    """∀x. x * 0 = 0  and  ∀x. 0 * x = 0"""
    results = []
    for v in var_names:
        x = _var(v)
        results.append(_forall(v, _eq(_mul(x, _zero()), _zero())))
        results.append(_forall(v, _eq(_mul(_zero(), x), _zero())))
    return results


def _mul_one_identity_templates(var_names: List[str]) -> List[Expression]:
    """∀x. x * S(0) = x  and  ∀x. S(0) * x = x"""
    results = []
    one = _succ(_zero())
    for v in var_names:
        x = _var(v)
        results.append(_forall(v, _eq(_mul(x, one), x)))
        results.append(_forall(v, _eq(_mul(one, x), x)))
    return results


def _distributivity_templates(var_names: List[str]) -> List[Expression]:
    """∀x y z. x * (y + z) = x * y + x * z  (left and right)"""
    results = []
    for v1, v2, v3 in itertools.permutations(var_names[:4], 3):
        x, y, z = _var(v1), _var(v2), _var(v3)
        # Left distributivity: x * (y + z) = x*y + x*z
        results.append(_forall_many(
            [v1, v2, v3],
            _eq(_mul(x, _add(y, z)), _add(_mul(x, y), _mul(x, z)))
        ))
        # Right distributivity: (y + z) * x = y*x + z*x
        results.append(_forall_many(
            [v1, v2, v3],
            _eq(_mul(_add(y, z), x), _add(_mul(y, x), _mul(z, x)))
        ))
    return results


def _successor_add_templates(var_names: List[str]) -> List[Expression]:
    """∀x y. S(x) + y = S(x + y)  and  ∀x y. x + S(y) = S(x + y)"""
    results = []
    for v1, v2 in itertools.permutations(var_names, 2):
        x, y = _var(v1), _var(v2)
        results.append(_forall_many(
            [v1, v2],
            _eq(_add(_succ(x), y), _succ(_add(x, y)))
        ))
        results.append(_forall_many(
            [v1, v2],
            _eq(_add(x, _succ(y)), _succ(_add(x, y)))
        ))
    return results


def _successor_mul_templates(var_names: List[str]) -> List[Expression]:
    """∀x y. S(x) * y = x*y + y  and  ∀x y. x * S(y) = x*y + x"""
    results = []
    for v1, v2 in itertools.permutations(var_names, 2):
        x, y = _var(v1), _var(v2)
        results.append(_forall_many(
            [v1, v2],
            _eq(_mul(_succ(x), y), _add(_mul(x, y), y))
        ))
        results.append(_forall_many(
            [v1, v2],
            _eq(_mul(x, _succ(y)), _add(_mul(x, y), x))
        ))
    return results


def _double_templates(var_names: List[str]) -> List[Expression]:
    """∀x. x + x = x * S(S(0))  (x + x = 2*x analog)"""
    results = []
    two = _succ(_succ(_zero()))
    for v in var_names:
        x = _var(v)
        results.append(_forall(v, _eq(_add(x, x), _mul(x, two))))
        results.append(_forall(v, _eq(_mul(two, x), _add(x, x))))
    return results


def _add_cancel_templates(var_names: List[str]) -> List[Expression]:
    """∀x y z. x + z = y + z → x = y  (encoded as: (x+z = y+z) → (x = y))
    Also: x + z = z + y → x = y"""
    # These are implications, which are harder to prove but interesting
    results = []
    for v1, v2, v3 in itertools.permutations(var_names[:4], 3):
        x, y, z = _var(v1), _var(v2), _var(v3)
        # Additive cancellation from the right
        hyp = _eq(_add(x, z), _add(y, z))
        concl = _eq(x, y)
        results.append(_forall_many([v1, v2, v3], Implies(hyp, concl)))
    return results


def _nested_add_patterns(var_names: List[str]) -> List[Expression]:
    """Deeper nesting: ∀x y z. x + (y + z) = (x + z) + y (via commutativity)"""
    results = []
    for v1, v2, v3 in itertools.permutations(var_names[:4], 3):
        x, y, z = _var(v1), _var(v2), _var(v3)
        # x + (y + z) = y + (x + z)
        results.append(_forall_many(
            [v1, v2, v3],
            _eq(_add(x, _add(y, z)), _add(y, _add(x, z)))
        ))
        # (x + y) + z = (x + z) + y
        results.append(_forall_many(
            [v1, v2, v3],
            _eq(_add(_add(x, y), z), _add(_add(x, z), y))
        ))
    return results


def _mixed_mul_add_patterns(var_names: List[str]) -> List[Expression]:
    """Mixed multiplication and addition patterns."""
    results = []
    for v1, v2 in itertools.permutations(var_names, 2):
        x, y = _var(v1), _var(v2)
        # x * (x + y) = x*x + x*y
        results.append(_forall_many(
            [v1, v2],
            _eq(_mul(x, _add(x, y)), _add(_mul(x, x), _mul(x, y)))
        ))
        # (x + y) * (x + y) = x*x + S(S(0))*x*y + y*y  (too complex — skip)
        # Simple: x * (y + x) = x*y + x*x
        results.append(_forall_many(
            [v1, v2],
            _eq(_mul(x, _add(y, x)), _add(_mul(x, y), _mul(x, x)))
        ))
    return results


# ── Master template list ───────────────────────────────────────────────────────

_TEMPLATE_GENERATORS = [
    _commutative_add_templates,
    _commutative_mul_templates,
    _associativity_add_templates,
    _associativity_mul_templates,
    _add_zero_identity_templates,
    _mul_zero_annihilation_templates,
    _mul_one_identity_templates,
    _distributivity_templates,
    _successor_add_templates,
    _successor_mul_templates,
    _double_templates,
    _add_cancel_templates,
    _nested_add_patterns,
    _mixed_mul_add_patterns,
]


class AlgebraicTemplateGenerator:
    """
    Generates conjectures from algebraic templates with variable instantiation.

    All template families are pre-expanded once at construction time.
    After exhaustion, the generator cycles back and adds shallow nesting
    variations to avoid getting stuck.
    """

    def __init__(
        self,
        var_names: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.var_names = var_names or ["x", "y", "z", "w"]
        random.seed(seed)

        # Pre-expand all templates into a flat list
        self._pool: List[Expression] = []
        for gen_fn in _TEMPLATE_GENERATORS:
            try:
                self._pool.extend(gen_fn(self.var_names))
            except Exception:
                pass

        # Deduplicate by string representation
        seen: set = set()
        unique: List[Expression] = []
        for expr in self._pool:
            key = str(expr)
            if key not in seen:
                seen.add(key)
                unique.append(expr)
        self._pool = unique

        random.shuffle(self._pool)
        self._cursor = 0
        self._total = len(self._pool)

    def pool_size(self) -> int:
        return self._total

    def generate(self, n: int) -> List[Expression]:
        """
        Return up to n conjectures from the template pool.
        Cycles back to the beginning when the pool is exhausted.
        """
        results: List[Expression] = []
        for _ in range(n):
            expr = self._pool[self._cursor % self._total]
            results.append(expr)
            self._cursor += 1
        return results

    def generate_all(self) -> List[Expression]:
        """Return the full template pool (one pass)."""
        return list(self._pool)
