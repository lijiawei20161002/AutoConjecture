"""
Proof engine that searches for proofs using tactics.
Uses best-first search with heuristics.

The engine supports multi-goal proof states: a ProofState carries a list
of Goal objects.  Tactics operate on goals[0] and may close it, transform
it, or split it into sub-goals.  The search terminates successfully when a
tactic application returns [] (all goals discharged).
"""
from __future__ import annotations
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import heapq

from ..logic.expressions import Expression, Forall, Implies
from ..logic.terms import Var, Zero, Succ
from .tactics import Tactic, ProofState, DEFAULT_TACTICS


class ProofResult(Enum):
    """Result of a proof attempt."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


@dataclass
class ProofStep:
    """A single step in a completed proof."""
    tactic: str
    state_before: ProofState
    state_after: Optional[ProofState]   # None → QED

    def __str__(self) -> str:
        if self.state_after is None:
            return f"  {self.tactic} → QED"
        return f"  {self.tactic} → {self.state_after.goal}"


@dataclass
class Proof:
    """A complete (or failed) proof of a statement."""
    statement: Expression
    steps: List[ProofStep]
    result: ProofResult

    def __str__(self) -> str:
        steps_str = "\n".join(str(s) for s in self.steps)
        return f"Proof of: {self.statement}\n{steps_str}\nResult: {self.result.value}"

    def length(self) -> int:
        return len(self.steps)


@dataclass
class SearchNode:
    """
    Node in the best-first search tree.
    Lower heuristic score = more promising.
    """
    state: ProofState
    parent: Optional["SearchNode"]
    tactic_used: Optional[str]
    heuristic_score: float

    def __lt__(self, other: "SearchNode") -> bool:
        return self.heuristic_score < other.heuristic_score


class ProofEngine:
    """
    Best-first proof search over ProofState nodes.

    Supports single-goal proofs (legacy) and multi-goal proofs introduced
    by InductionTactic and other structural tactics.
    """

    def __init__(
        self,
        tactics: Optional[List[Tactic]] = None,
        max_depth: int = 50,
        max_iterations: int = 1000,
        knowledge_base: Optional[List[Expression]] = None,
    ):
        self.tactics       = tactics if tactics is not None else DEFAULT_TACTICS
        self.max_depth     = max_depth
        self.max_iterations = max_iterations
        self.knowledge_base = knowledge_base if knowledge_base is not None else []

    def prove(
        self,
        goal: Expression,
        hypotheses: Optional[List[Expression]] = None,
    ) -> Proof:
        """
        Attempt to prove `goal` given `hypotheses`.

        Strategy:
          1. For ∀x.P(x) goals, try the induction strategy first.
             It splits into base + step and proves each with a fresh BFS call,
             so budget is not wasted mixing inductive and non-inductive states.
          2. Fall back to flat BFS for everything else (and for induction
             failures, so the BFS can still find direct / axiom proofs).
        """
        hyps = hypotheses if hypotheses is not None else []

        # Induction pre-pass for Forall goals (up to max_induction_depth layers)
        if isinstance(goal, Forall):
            ind = self._try_induction(goal, hyps, depth=0)
            if ind is not None and ind.result == ProofResult.SUCCESS:
                return ind

        return self._bfs(goal, hyps)

    # ── induction strategy ────────────────────────────────────────────────

    def _try_induction(
        self,
        goal: Expression,
        hypotheses: List[Expression],
        depth: int,
        max_depth: int = 3,
    ) -> Optional[Proof]:
        """
        Prove ∀x.P(x) by structural induction on x.

        Recursively handles nested Forall goals (e.g. ∀x.∀y.P(x,y)) up to
        max_depth layers of induction.

        Returns a Proof on success, None if induction does not apply or
        either subgoal cannot be proved.
        """
        if not isinstance(goal, Forall) or depth >= max_depth:
            return None

        var_name = goal.var.name
        body     = goal.body

        # ── base case: P(0) ───────────────────────────────────────────
        base_expr = body.substitute(var_name, Zero())
        base_proof = self._prove_subgoal(base_expr, hypotheses)
        if base_proof.result != ProofResult.SUCCESS:
            return None

        # ── inductive step: ∀k. P(k) → P(S(k)) ──────────────────────
        k_name = self._fresh_var(var_name, body)
        k_var  = Var(k_name)
        ih     = body.substitute(var_name, k_var)           # P(k)
        step_goal_expr = body.substitute(var_name, Succ(k_var))  # P(S(k))
        step_hyps      = list(hypotheses) + [ih]

        # Step goal may itself be a Forall (e.g., ∀y. ...) — try nested induction
        if isinstance(step_goal_expr, Forall):
            step_proof = self._try_induction(
                step_goal_expr, step_hyps, depth + 1, max_depth
            )
            if step_proof is None or step_proof.result != ProofResult.SUCCESS:
                step_proof = self._prove_subgoal(step_goal_expr, step_hyps)
        else:
            step_proof = self._prove_subgoal(step_goal_expr, step_hyps)

        if step_proof.result != ProofResult.SUCCESS:
            return None

        # ── combine ───────────────────────────────────────────────────
        combined_steps = (
            [ProofStep("induction", ProofState(goal=goal, hypotheses=hypotheses), None)]
            + base_proof.steps
            + step_proof.steps
        )
        return Proof(statement=goal, steps=combined_steps,
                     result=ProofResult.SUCCESS)

    def _prove_subgoal(
        self,
        goal: Expression,
        hypotheses: List[Expression],
    ) -> Proof:
        """Prove a subgoal, trying induction if applicable."""
        if isinstance(goal, Forall):
            ind = self._try_induction(goal, hypotheses, depth=0)
            if ind is not None and ind.result == ProofResult.SUCCESS:
                return ind
        return self._bfs(goal, hypotheses)

    def _fresh_var(self, base: str, body: Expression) -> str:
        """Return a variable name not free in body."""
        used = body.free_vars()
        candidate = base + "0"
        while candidate in used:
            candidate += "0"
        return candidate

    # ── BFS ───────────────────────────────────────────────────────────────

    def _bfs(
        self,
        goal: Expression,
        hypotheses: List[Expression],
    ) -> Proof:
        """Flat best-first search for a proof of `goal`."""
        initial_state = ProofState(goal=goal, hypotheses=hypotheses, depth=0)

        pq: List[Tuple[float, int, SearchNode]] = []
        counter = 0

        root = SearchNode(
            state=initial_state,
            parent=None,
            tactic_used=None,
            heuristic_score=self._heuristic(initial_state),
        )
        heapq.heappush(pq, (root.heuristic_score, counter, root))
        counter += 1

        visited: set = set()
        iterations = 0

        while pq and iterations < self.max_iterations:
            iterations += 1
            _, _, current = heapq.heappop(pq)
            state = current.state

            key = state.key()
            if key in visited:
                continue
            visited.add(key)

            if state.depth > self.max_depth:
                continue

            for tactic in self.tactics:
                new_states = tactic.apply(state, self.knowledge_base)

                if len(new_states) == 0:
                    steps = self._reconstruct(current, tactic)
                    return Proof(statement=goal, steps=steps,
                                 result=ProofResult.SUCCESS)

                for ns in new_states:
                    if ns.key() == key:
                        continue
                    node = SearchNode(
                        state=ns,
                        parent=current,
                        tactic_used=tactic.name(),
                        heuristic_score=self._heuristic(ns),
                    )
                    heapq.heappush(pq, (node.heuristic_score, counter, node))
                    counter += 1

        result = (ProofResult.TIMEOUT if iterations >= self.max_iterations
                  else ProofResult.FAILURE)
        return Proof(statement=goal, steps=[], result=result)

    # ── heuristic ─────────────────────────────────────────────────────────

    def _heuristic(self, state: ProofState) -> float:
        """
        Lower score = more promising.

        Design choices:
        - Score only goals[0] (current focus) so that InductionTactic is not
          penalised for creating a large-looking step goal not yet attempted.
        - Forall goals are inflated by FORALL_PENALTY per quantifier layer.
          This makes applying InductionTactic (which exposes an Equation base
          case of lower apparent size) score better than doing IntroTactic
          repeatedly on a still-Forall goal.
        - Small flat cost per pending obligation and per depth.
        """
        FORALL_PENALTY = 8.0   # per ∀ layer still in the current goal

        if state.is_complete:
            return 0.0
        expr = state.goals[0].expression
        base_size = expr.size()

        # Count outer Forall layers and add penalty
        layers, inner = 0, expr
        while isinstance(inner, Forall):
            layers += 1
            inner = inner.body
        forall_pen = layers * FORALL_PENALTY

        pending_pen = (len(state.goals) - 1) * 3.0
        depth_pen   = state.depth * 0.2
        return base_size + forall_pen + pending_pen + depth_pen

    # ── proof reconstruction ──────────────────────────────────────────────

    def _reconstruct(self, final_node: SearchNode,
                     final_tactic: Tactic) -> List[ProofStep]:
        steps: List[ProofStep] = []
        current = final_node
        while current.parent is not None:
            steps.append(ProofStep(
                tactic=current.tactic_used,
                state_before=current.parent.state,
                state_after=current.state,
            ))
            current = current.parent
        steps.append(ProofStep(
            tactic=final_tactic.name(),
            state_before=final_node.state,
            state_after=None,
        ))
        steps.reverse()
        return steps

    # ── knowledge base helpers ────────────────────────────────────────────

    def add_to_knowledge_base(self, theorem: Expression) -> None:
        self.knowledge_base.append(theorem)

    def get_knowledge_base_size(self) -> int:
        return len(self.knowledge_base)
