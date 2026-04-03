"""
Tactics for theorem proving.
A tactic transforms a proof state by applying logical rules.

Tactics:
- Reflexivity:  prove t = t
- Assumption:   close goal if it matches a hypothesis
- Simplify:     reduce goal via arithmetic identities
- Rewrite:      replace a subterm using an equation from hypotheses / KB
- Substitute:   instantiate a Forall with a specific term
- Intro:        peel off a Forall, introducing the variable as free
- Implies:      for A → B goals, add A as a hypothesis and focus on B
- Induction:    for ∀x.P(x), split into base case P(0) and inductive
                step ∀k. P(k) → P(S(k))

ProofState now holds a *list* of goals (a proof-obligation queue).  The
current focus is always goals[0].  A tactic that closes goals[0] returns
a state with goals[1:]; a tactic that is not applicable returns the state
unchanged.  Returning [] signals that every goal has been discharged (QED).

Backward compatibility: ProofState(goal=e, hypotheses=h, depth=d) still works.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from ..logic.terms import Term, Var, Zero, Succ, Add, Mul
from ..logic.expressions import (
    Expression, Equation, Forall, Exists, And, Or, Implies, Not
)


# ─────────────────────────────────────────────────────────────────────────────
# Goal: one proof obligation with its local hypotheses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Goal:
    """
    A single proof obligation: an expression to prove given local hypotheses.
    """
    expression: Expression
    hypotheses: List[Expression] = field(default_factory=list)

    def __str__(self) -> str:
        hyps = ", ".join(str(h) for h in self.hypotheses) if self.hypotheses else "∅"
        return f"[{hyps}] ⊢ {self.expression}"


# ─────────────────────────────────────────────────────────────────────────────
# ProofState
# ─────────────────────────────────────────────────────────────────────────────

class ProofState:
    """
    A list of goals remaining to prove.  goals[0] is the current focus.

    Backward-compatible constructor:
        ProofState(goal=e, hypotheses=h, depth=d)   # old single-goal form
        ProofState(goals=[Goal(...),...], depth=d)   # new multi-goal form
    """

    def __init__(
        self,
        goal: Optional[Expression] = None,
        hypotheses: Optional[List[Expression]] = None,
        depth: int = 0,
        goals: Optional[List[Goal]] = None,
    ):
        if goals is not None:
            self.goals: List[Goal] = goals
        elif goal is not None:
            self.goals = [Goal(goal, list(hypotheses or []))]
        else:
            self.goals = []
        self.depth = depth

    # ── convenience properties (backward compat) ──────────────────────────

    @property
    def goal(self) -> Optional[Expression]:
        return self.goals[0].expression if self.goals else None

    @property
    def hypotheses(self) -> List[Expression]:
        return self.goals[0].hypotheses if self.goals else []

    @property
    def is_complete(self) -> bool:
        return len(self.goals) == 0

    # ── helpers used by tactics ───────────────────────────────────────────

    def close_current(self) -> List["ProofState"]:
        """Return the state after closing goals[0].  [] if that was the last."""
        remaining = self.goals[1:]
        if not remaining:
            return []
        return [ProofState(goals=remaining, depth=self.depth + 1)]

    def replace_current(self, new_expr: Expression,
                        new_hyps: Optional[List[Expression]] = None) -> "ProofState":
        """Replace goals[0] with a new goal (same hypotheses by default)."""
        hyps = new_hyps if new_hyps is not None else list(self.hypotheses)
        return ProofState(
            goals=[Goal(new_expr, hyps)] + self.goals[1:],
            depth=self.depth + 1,
        )

    def prepend_goals(self, new_goals: List[Goal]) -> "ProofState":
        """Insert new_goals before the remaining goals (replaces goals[0])."""
        return ProofState(
            goals=new_goals + self.goals[1:],
            depth=self.depth + 1,
        )

    # ── display / hashing ────────────────────────────────────────────────

    def key(self) -> str:
        """Unique key for visited-set deduplication (includes hypotheses)."""
        parts = []
        for g in self.goals:
            hyps = ";".join(sorted(str(h) for h in g.hypotheses))
            parts.append(f"[{hyps}]⊢{g.expression}")
        return "|".join(parts)

    def __str__(self) -> str:
        if self.is_complete:
            return "QED"
        lines = [f"Goal ({len(self.goals)} obligation(s)):"]
        for i, g in enumerate(self.goals):
            marker = "▶" if i == 0 else " "
            hyps = ", ".join(str(h) for h in g.hypotheses) or "∅"
            lines.append(f"  {marker} [{hyps}] ⊢ {g.expression}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tactic base class
# ─────────────────────────────────────────────────────────────────────────────

class Tactic(ABC):
    """
    A tactic transforms one ProofState into zero or more new ProofStates.
    Returning [] signals QED (all goals discharged).
    """

    @abstractmethod
    def apply(self, state: ProofState,
              knowledge_base: List[Expression]) -> List[ProofState]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Core tactics
# ─────────────────────────────────────────────────────────────────────────────

class ReflexivityTactic(Tactic):
    """Close goal if it is of the form t = t."""

    def apply(self, state: ProofState,
              knowledge_base: List[Expression]) -> List[ProofState]:
        if not state.goals:
            return []
        goal = state.goals[0].expression
        if isinstance(goal, Equation):
            if str(goal.left) == str(goal.right):
                return state.close_current()
        return [state]

    def name(self) -> str:
        return "reflexivity"


class AssumptionTactic(Tactic):
    """Close goal if it syntactically matches a local hypothesis or KB entry."""

    def apply(self, state: ProofState,
              knowledge_base: List[Expression]) -> List[ProofState]:
        if not state.goals:
            return []
        goal_str = str(state.goals[0].expression)
        for src in (state.goals[0].hypotheses, knowledge_base):
            for stmt in src:
                if str(stmt) == goal_str:
                    return state.close_current()
        return [state]

    def name(self) -> str:
        return "assumption"


class SimplifyTactic(Tactic):
    """
    Apply Peano arithmetic simplification rules:
      0 + t = t,   t + 0 = t,   0 * t = 0,   t * 0 = 0
    Close the goal if both sides reduce to the same normal form.
    """

    def apply(self, state: ProofState,
              knowledge_base: List[Expression]) -> List[ProofState]:
        if not state.goals:
            return []
        goal = state.goals[0].expression
        if not isinstance(goal, Equation):
            return [state]

        left_s  = self._simplify(goal.left)
        right_s = self._simplify(goal.right)

        if str(left_s) == str(right_s):
            return state.close_current()

        if str(left_s) != str(goal.left) or str(right_s) != str(goal.right):
            return [state.replace_current(Equation(left_s, right_s))]

        return [state]

    def _simplify(self, term: Term) -> Term:
        if isinstance(term, Add):
            left  = self._simplify(term.left)
            right = self._simplify(term.right)
            if isinstance(left, Zero):
                return right
            if isinstance(right, Zero):
                return left
            return Add(left, right)
        elif isinstance(term, Mul):
            left  = self._simplify(term.left)
            right = self._simplify(term.right)
            if isinstance(left, Zero) or isinstance(right, Zero):
                return Zero()
            return Mul(left, right)
        elif isinstance(term, Succ):
            return Succ(self._simplify(term.term))
        return term

    def name(self) -> str:
        return "simplify"


class RewriteTactic(Tactic):
    """
    Rewrite the current goal using equations from local hypotheses or KB.

    Handles two cases:
      - Ground equations (no free variables): string-equality matching.
      - Universally quantified equations ∀x.∀y. lhs=rhs: unification-based
        matching, so that e.g. x+S(y)=S(x+y) can rewrite S(0)+S(k).

    Both top-level and subterm rewrites are attempted in both directions.
    """

    def apply(self, state: ProofState,
              knowledge_base: List[Expression]) -> List[ProofState]:
        if not state.goals:
            return []
        goal = state.goals[0].expression
        if not isinstance(goal, Equation):
            return [state]

        new_states: List[ProofState] = []
        gl, gr = goal.left, goal.right

        for src in (state.goals[0].hypotheses, knowledge_base):
            for stmt in src:
                eq_and_vars = self._extract(stmt)
                if eq_and_vars is None:
                    continue
                eq, pattern_vars = eq_and_vars
                for new_goal in self._rewrites(gl, gr, eq, pattern_vars):
                    ns = state.replace_current(new_goal)
                    new_states.append(ns)

        return new_states if new_states else [state]

    # ── extraction ────────────────────────────────────────────────────

    def _extract(self, stmt: Expression):
        """Return (Equation, set_of_bound_var_names) after peeling Foralls."""
        bound: List[str] = []
        while isinstance(stmt, Forall):
            bound.append(stmt.var.name)
            stmt = stmt.body
        if isinstance(stmt, Equation):
            return stmt, set(bound)
        return None

    # ── rewrite generation ────────────────────────────────────────────

    def _rewrites(self, gl: Term, gr: Term,
                  eq: "Equation", pvars: Set[str]) -> List["Equation"]:
        """
        Generate all rewritten goals for both directions of eq.
        Uses unification when pvars is non-empty, string match otherwise.
        """
        results: List[Equation] = []
        el, er = eq.left, eq.right

        for lhs, rhs in [(el, er), (er, el)]:
            # Top-level full match
            for new_gl in self._apply_top(gl, lhs, rhs, pvars):
                results.append(Equation(new_gl, gr))
            for new_gr in self._apply_top(gr, lhs, rhs, pvars):
                results.append(Equation(gl, new_gr))
            # Subterm match
            for new_gl in self._apply_sub(gl, lhs, rhs, pvars):
                if str(new_gl) != str(gl):
                    results.append(Equation(new_gl, gr))
            for new_gr in self._apply_sub(gr, lhs, rhs, pvars):
                if str(new_gr) != str(gr):
                    results.append(Equation(gl, new_gr))

        return results

    def _apply_top(self, t: Term, lhs: Term, rhs: Term,
                   pvars: Set[str]) -> List[Term]:
        """Return [rhs_instantiated] if lhs unifies with t, else []."""
        subst = self._unify(lhs, t, pvars)
        if subst is None:
            return []
        return [self._apply_subst(rhs, subst)]

    def _apply_sub(self, t: Term, lhs: Term, rhs: Term,
                   pvars: Set[str]) -> List[Term]:
        """Try lhs→rhs rewrite on every subterm of t; return all rewrites."""
        results: List[Term] = []
        if isinstance(t, Succ):
            for inner in self._apply_sub(t.term, lhs, rhs, pvars):
                results.append(Succ(inner))
            for top in self._apply_top(t.term, lhs, rhs, pvars):
                results.append(Succ(top))
        elif isinstance(t, Add):
            for l2 in self._apply_sub(t.left, lhs, rhs, pvars):
                results.append(Add(l2, t.right))
            for l2 in self._apply_top(t.left, lhs, rhs, pvars):
                results.append(Add(l2, t.right))
            for r2 in self._apply_sub(t.right, lhs, rhs, pvars):
                results.append(Add(t.left, r2))
            for r2 in self._apply_top(t.right, lhs, rhs, pvars):
                results.append(Add(t.left, r2))
        elif isinstance(t, Mul):
            for l2 in self._apply_sub(t.left, lhs, rhs, pvars):
                results.append(Mul(l2, t.right))
            for l2 in self._apply_top(t.left, lhs, rhs, pvars):
                results.append(Mul(l2, t.right))
            for r2 in self._apply_sub(t.right, lhs, rhs, pvars):
                results.append(Mul(t.left, r2))
            for r2 in self._apply_top(t.right, lhs, rhs, pvars):
                results.append(Mul(t.left, r2))
        return results

    # ── unification ───────────────────────────────────────────────────

    def _unify(self, pattern: Term, target: Term,
               pvars: Set[str]) -> Optional[Dict[str, Term]]:
        """
        Unify pattern against target, treating names in pvars as variables.
        Returns a substitution dict or None on failure.
        """
        subst: Dict[str, Term] = {}
        if self._unify_rec(pattern, target, pvars, subst):
            return subst
        return None

    def _unify_rec(self, p: Term, t: Term,
                   pvars: Set[str], subst: Dict[str, Term]) -> bool:
        if isinstance(p, Var) and p.name in pvars:
            if p.name in subst:
                return str(subst[p.name]) == str(t)
            subst[p.name] = t
            return True
        if type(p) is not type(t):
            return False
        if isinstance(p, (Var, Zero)):
            return str(p) == str(t)
        if isinstance(p, Succ):
            return self._unify_rec(p.term, t.term, pvars, subst)
        if isinstance(p, Add):
            return (self._unify_rec(p.left,  t.left,  pvars, subst) and
                    self._unify_rec(p.right, t.right, pvars, subst))
        if isinstance(p, Mul):
            return (self._unify_rec(p.left,  t.left,  pvars, subst) and
                    self._unify_rec(p.right, t.right, pvars, subst))
        return False

    def _apply_subst(self, t: Term, subst: Dict[str, Term]) -> Term:
        """Apply substitution to a term."""
        if isinstance(t, Var) and t.name in subst:
            return subst[t.name]
        if isinstance(t, Succ):
            return Succ(self._apply_subst(t.term, subst))
        if isinstance(t, Add):
            return Add(self._apply_subst(t.left, subst),
                       self._apply_subst(t.right, subst))
        if isinstance(t, Mul):
            return Mul(self._apply_subst(t.left, subst),
                       self._apply_subst(t.right, subst))
        return t

    def name(self) -> str:
        return "rewrite"


class SubstituteTactic(Tactic):
    """
    Instantiate a Forall goal with a specific term.
    SubstituteTactic(var, term): replaces ∀var.P(var) with P(term).
    """

    def __init__(self, var: Var, term: Term):
        self.var  = var
        self.term = term

    def apply(self, state: ProofState,
              knowledge_base: List[Expression]) -> List[ProofState]:
        if not state.goals:
            return []
        goal = state.goals[0].expression
        if isinstance(goal, Forall):
            new_body = goal.body.substitute(goal.var.name, self.term)
            return [state.replace_current(new_body)]
        return [state]

    def name(self) -> str:
        return f"substitute({self.var.name} := {self.term})"


# ─────────────────────────────────────────────────────────────────────────────
# Structural tactics (new)
# ─────────────────────────────────────────────────────────────────────────────

class IntroTactic(Tactic):
    """
    For a goal of the form ∀x.P(x), peel off the quantifier and focus on
    P(x) with x treated as an arbitrary free variable.

    This is the standard "introduction rule" for universal statements and is
    needed to set up the context for the inductive step.
    """

    def apply(self, state: ProofState,
              knowledge_base: List[Expression]) -> List[ProofState]:
        if not state.goals:
            return []
        goal = state.goals[0].expression
        if not isinstance(goal, Forall):
            return [state]
        # Expose the body; the bound variable becomes free in the new goal.
        return [state.replace_current(goal.body)]

    def name(self) -> str:
        return "intro"


class ImpliesTactic(Tactic):
    """
    For a goal of the form A → B, add A to the local hypotheses and focus
    on B.  This gives the induction hypothesis during the inductive step.
    """

    def apply(self, state: ProofState,
              knowledge_base: List[Expression]) -> List[ProofState]:
        if not state.goals:
            return []
        goal = state.goals[0].expression
        if not isinstance(goal, Implies):
            return [state]
        new_hyps = list(state.goals[0].hypotheses) + [goal.hypothesis]
        return [state.replace_current(goal.conclusion, new_hyps)]

    def name(self) -> str:
        return "intro_implies"


class InductionTactic(Tactic):
    """
    Peano induction for a goal of the form ∀x. P(x).

    Splits into two subgoals pushed onto the front of the obligation queue:
      1. Base case:      P(0)
      2. Inductive step: ∀k. P(k) → P(S(k))

    The search then proves each subgoal in turn.  The inductive hypothesis
    P(k) is available as a local hypothesis once IntroTactic + ImpliesTactic
    have been applied to the step goal.
    """

    def apply(self, state: ProofState,
              knowledge_base: List[Expression]) -> List[ProofState]:
        if not state.goals:
            return []
        goal = state.goals[0].expression
        if not isinstance(goal, Forall):
            return [state]

        var_name = goal.var.name
        body     = goal.body

        # Base case: P(0)
        base_expr = body.substitute(var_name, Zero())

        # Inductive step: choose a fresh variable name k
        k_name = self._fresh(var_name, body)
        k_var  = Var(k_name)
        ih     = body.substitute(var_name, k_var)            # P(k)
        concl  = body.substitute(var_name, Succ(k_var))      # P(S(k))
        step_expr = Forall(k_var, Implies(ih, concl))

        base_goal = Goal(base_expr,  list(state.goals[0].hypotheses))
        step_goal = Goal(step_expr,  list(state.goals[0].hypotheses))

        return [state.prepend_goals([base_goal, step_goal])]

    def _fresh(self, base: str, body: Expression) -> str:
        """Return a variable name not free in body."""
        used = body.free_vars()
        candidate = base + "0"
        while candidate in used:
            candidate += "0"
        return candidate

    def name(self) -> str:
        return "induction"


# ─────────────────────────────────────────────────────────────────────────────
# Default tactic portfolio
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TACTICS: List[Tactic] = [
    ReflexivityTactic(),
    AssumptionTactic(),
    SimplifyTactic(),
    RewriteTactic(),
    IntroTactic(),
    ImpliesTactic(),
    InductionTactic(),
]
