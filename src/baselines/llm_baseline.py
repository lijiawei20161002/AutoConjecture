"""
Baseline 6: LLM-guided conjecture generation (GPT-4o) + heuristic BFS prover.

Analogous to FunSearch-light: GPT-4o sees the current knowledge base and
proposes new conjectures; the existing ProofEngine verifies them.  No RL,
no neural prover — the sole learning signal is the growing KB fed back as
context to the LLM.

Requires:
    OPENAI_API_KEY in environment (or .env file at project root / Desktop).

Conjectures are generated in a safe Python-constructor format, e.g.
    Forall(Var("x"), Equation(Add(Var("x"), Zero()), Var("x")))
which is eval'd in a restricted namespace containing only the AST node types.
"""
from __future__ import annotations

import json
import os
import time
from typing import List, Optional

from ..logic.axioms import get_all_axioms
from ..logic.terms import Var, Zero, Succ, Add, Mul
from ..logic.expressions import Expression, Equation, Forall, Exists, And, Or, Not, Implies
from ..knowledge.knowledge_base import KnowledgeBase
from ..prover.proof_engine import ProofEngine, ProofResult
from ..generation.heuristics import ComplexityEstimator, DiversityFilter
from ..generation.novelty import NoveltyScorer
from ..comparison.metrics import ComparisonMetrics, ComparisonSnapshot
from .base_runner import BaselineRunner

# ── safe eval namespace ────────────────────────────────────────────────────────
_SAFE_NS = {
    "Var": Var, "Zero": Zero, "Succ": Succ, "Add": Add, "Mul": Mul,
    "Equation": Equation, "Forall": Forall, "Exists": Exists,
    "And": And, "Or": Or, "Not": Not, "Implies": Implies,
    "__builtins__": {},
}

# ── system prompt ──────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an expert in Peano arithmetic. Your task is to conjecture new theorems.

You work with these expression constructors (Python syntax):
  Terms:       Var("x"), Zero(), Succ(t), Add(t1,t2), Mul(t1,t2)
  Expressions: Equation(lhs, rhs), Forall(Var("x"), body),
               Exists(Var("x"), body), And(e1,e2), Or(e1,e2),
               Not(e), Implies(hyp, concl)

Variables available: x, y, z, w.

Rules:
- Every top-level expression MUST be wrapped in at least one Forall.
- Only output valid Python constructor expressions, one per line.
- Do NOT include any explanation, comments, or import statements.
- Aim for non-trivial, diverse conjectures — avoid restating known theorems.
- Complexity: 6–20 AST nodes. Prefer universal statements about +, *, Succ.

Example valid conjectures:
Forall(Var("x"), Equation(Add(Var("x"), Zero()), Var("x")))
Forall(Var("x"), Forall(Var("y"), Equation(Add(Var("x"), Var("y")), Add(Var("y"), Var("x")))))
Forall(Var("x"), Equation(Mul(Var("x"), Succ(Zero())), Var("x")))
"""


def _load_api_key() -> str:
    """Load OPENAI_API_KEY from environment or .env files."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    # Try project .env, then Desktop .env
    for path in [
        os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        os.path.expanduser("~/Desktop/.env"),
    ]:
        path = os.path.abspath(path)
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(
        "OPENAI_API_KEY not found. Set it in the environment or in .env"
    )


def _safe_parse(expr_str: str) -> Optional[Expression]:
    """Parse a constructor-syntax string into an Expression, or return None."""
    expr_str = expr_str.strip()
    if not expr_str:
        return None
    try:
        result = eval(expr_str, _SAFE_NS)  # noqa: S307
        if isinstance(result, Expression):
            return result
    except Exception:
        pass
    return None


class LLMBaselineRunner(BaselineRunner):
    """
    GPT-4o conjecture generator + heuristic BFS prover.

    GPT-4o sees the current KB at each call and proposes new conjectures.
    Proven theorems are added to the KB and fed back at the next call.
    """

    @property
    def name(self) -> str:
        return "llm_gpt4o"

    def setup(self, config: dict) -> None:
        self.kb = KnowledgeBase(axioms=get_all_axioms())

        self._api_key = _load_api_key()
        self._model = config.get("llm_model", "gpt-4o")
        self._batch_size = config.get("llm_batch_size", 10)
        self._max_tokens = config.get("llm_max_tokens", 800)
        self._temperature = config.get("llm_temperature", 1.0)
        self._retry_delay = config.get("llm_retry_delay", 2.0)

        var_names = config.get("var_names", ["x", "y", "z", "w"])
        self.engine = ProofEngine(
            max_depth=config.get("max_depth", 50),
            max_iterations=config.get("max_iterations", 500),
            knowledge_base=self.kb.get_all_statements(),
        )
        self.complexity_est = ComplexityEstimator()
        self.diversity_filter = DiversityFilter(max_similar=config.get("max_similar", 50))
        self.novelty_scorer = NoveltyScorer()
        self.metrics = ComparisonMetrics(self.name)

        self._attempted = 0
        self._proved = 0
        self._llm_calls = 0
        self._parse_failures = 0

        # Live log: every proven theorem written here immediately
        self._live_log_path: Optional[str] = config.get("live_log_path")
        if self._live_log_path:
            os.makedirs(os.path.dirname(self._live_log_path) or ".", exist_ok=True)

        print(f"[{self.name}] model={self._model}, batch={self._batch_size}", flush=True)

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

            # ── generate via LLM ──────────────────────────────────────────────
            conjectures = self._call_llm()

            # ── filter ────────────────────────────────────────────────────────
            kb_stmts = self.kb.get_all_statements()
            self.engine.knowledge_base = kb_stmts

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

            # ── prove ─────────────────────────────────────────────────────────
            for c in filtered:
                self._attempted += 1
                proof = self.engine.prove(c)
                if proof.result == ProofResult.SUCCESS:
                    complexity = self.complexity_est.estimate(c)
                    added = self.kb.add_theorem(c, proof, complexity, 0, self._attempted)
                    if added:
                        self._proved += 1
                        self.novelty_scorer.add(c)
                        self._log_theorem(c, proof, time.time() - t_start)

            # ── snapshot ──────────────────────────────────────────────────────
            now = time.time()
            if (now - t_last_snap) >= snapshot_interval_seconds:
                snap = self.metrics.snapshot(
                    self.kb, now - t_start, self._attempted, self._proved
                )
                snapshots.append(snap)
                print(
                    f"{snap.summary_line()} | llm_calls={self._llm_calls}"
                    f" parse_fail={self._parse_failures}",
                    flush=True,
                )
                t_last_snap = now

        final_snap = self.metrics.snapshot(
            self.kb, time.time() - t_start, self._attempted, self._proved
        )
        snapshots.append(final_snap)
        return snapshots

    def get_final_kb(self) -> KnowledgeBase:
        return self.kb

    # ── private ────────────────────────────────────────────────────────────────

    def _call_llm(self) -> List[Optional[Expression]]:
        """Ask GPT-4o to generate self._batch_size new conjectures."""
        import openai  # lazy import so the class can be imported without openai

        kb_theorems = self.kb.get_all_statements()
        known_strs = "\n".join(f"  {str(t)}" for t in list(kb_theorems)[-20:])
        user_msg = (
            f"Known theorems in KB (last 20):\n{known_strs}\n\n"
            f"Generate {self._batch_size} NEW conjectures not already in the KB. "
            "Output one constructor expression per line, nothing else."
        )

        for attempt in range(3):
            try:
                client = openai.OpenAI(api_key=self._api_key)
                resp = client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                self._llm_calls += 1
                raw = resp.choices[0].message.content or ""
                return self._parse_response(raw)
            except Exception as e:
                print(f"[{self.name}] LLM call failed (attempt {attempt+1}): {e}", flush=True)
                time.sleep(self._retry_delay * (attempt + 1))

        return []

    def _parse_response(self, raw: str) -> List[Optional[Expression]]:
        """Parse one-expression-per-line GPT-4o output."""
        results = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            expr = _safe_parse(line)
            if expr is None:
                self._parse_failures += 1
            results.append(expr)
        return results

    def _log_theorem(self, expr: Expression, proof, elapsed: float) -> None:
        """Append a proven theorem to the live JSONL log."""
        if not self._live_log_path:
            return
        entry = {
            "system": self.name,
            "elapsed_s": round(elapsed, 2),
            "theorem": str(expr),
            "proof_steps": getattr(proof, "steps", None),
            "complexity": self.complexity_est.estimate(expr),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self._live_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
