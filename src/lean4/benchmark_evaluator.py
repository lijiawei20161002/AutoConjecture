"""
Benchmark evaluator for Lean 4 integration.

Supports:
  - MiniF2F  (244 test problems, Lean 4 official format)
  - LeanWorkbook (subset, JSONL format)
  - ProofNet (undergraduate math, Lean 4 subset)
  - Open-ended discovery (generate + prove novel Lean 4 theorems)

Each benchmark returns a BenchmarkResults object with standard metrics.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .repl_interface import Lean4REPLInterface, Lean4NotAvailable
from .lean4_prover import Lean4TacticProver, Lean4ProofResult
from .lean4_generator import Lean4ConjectureGenerator


@dataclass
class BenchmarkProblem:
    """One problem from a standard benchmark."""
    name: str
    formal_statement: str    # Lean 4 theorem declaration (body only, no proof)
    informal_statement: Optional[str] = None
    split: str = "test"      # "train" / "valid" / "test"
    category: Optional[str] = None    # e.g. "algebra", "number_theory"
    source: Optional[str] = None      # e.g. "miniF2F", "putnam"


@dataclass
class BenchmarkResults:
    """Results of evaluating a system on one benchmark."""
    benchmark_name: str
    total_problems: int
    solved: int
    pass_at_1: float
    pass_at_k: Dict[int, float] = field(default_factory=dict)
    avg_proof_time_seconds: float = 0.0
    solved_by_tactic: Dict[str, int] = field(default_factory=dict)
    unsolved_names: List[str] = field(default_factory=list)
    solved_names: List[str] = field(default_factory=list)
    # Maps problem name → {statement, tactic, proof_time_s}
    solved_details: Dict[str, dict] = field(default_factory=dict)
    evaluation_time_seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Benchmark : {self.benchmark_name}",
            f"Problems  : {self.total_problems}",
            f"Solved    : {self.solved}  ({self.pass_at_1:.2%})",
            f"Avg time  : {self.avg_proof_time_seconds:.2f}s per problem",
        ]
        if self.pass_at_k:
            for k, v in sorted(self.pass_at_k.items()):
                lines.append(f"Pass@{k:3d}  : {v:.2%}")
        if self.solved_by_tactic:
            lines.append("By tactic:")
            for tac, cnt in sorted(self.solved_by_tactic.items(), key=lambda x: -x[1]):
                lines.append(f"  {tac}: {cnt}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark_name,
            "total": self.total_problems,
            "solved": self.solved,
            "pass_at_1": round(self.pass_at_1, 4),
            "pass_at_k": {str(k): round(v, 4) for k, v in self.pass_at_k.items()},
            "avg_proof_time_s": round(self.avg_proof_time_seconds, 3),
            "solved_by_tactic": dict(self.solved_by_tactic),
            "solved_names": self.solved_names,
            "unsolved_names": self.unsolved_names,
            "solved_details": self.solved_details,
            "eval_time_s": round(self.evaluation_time_seconds, 1),
        }


class BenchmarkEvaluator:
    """
    Evaluates AutoConjecture's Lean 4 system on standard benchmarks.

    Usage:
        with Lean4REPLInterface(mode="oneshot") as repl:
            prover = Lean4TacticProver(repl)
            gen = Lean4ConjectureGenerator()
            evaluator = BenchmarkEvaluator(repl, prover, gen)

            problems = evaluator.load_minif2f("/path/to/minif2f/lean4")
            results = evaluator.evaluate(problems, benchmark_name="miniF2F-test")
            print(results.summary())
    """

    def __init__(
        self,
        repl: Lean4REPLInterface,
        prover: Lean4TacticProver,
        generator: Optional[Lean4ConjectureGenerator] = None,
    ):
        self.repl = repl
        self.prover = prover
        self.generator = generator

    # ── Loaders ────────────────────────────────────────────────────────────────

    def load_minif2f(self, data_dir: str, split: str = "test") -> List[BenchmarkProblem]:
        """
        Load MiniF2F problems from a local directory.

        Expected directory layout (from the official MiniF2F repo):
            <data_dir>/
              test/  or  valid/
                algebra/  number_theory/  ...
                  *.lean   (one file per problem, named <problem_name>.lean)

        Or a JSONL file at <data_dir>/minif2f_lean4.jsonl with fields:
          {"name": str, "formal_statement": str, "split": str}
        """
        problems: List[BenchmarkProblem] = []

        # Try JSONL format first
        jsonl_path = os.path.join(data_dir, "minif2f_lean4.jsonl")
        if os.path.isfile(jsonl_path):
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    if d.get("split", "test") == split:
                        problems.append(BenchmarkProblem(
                            name=d["name"],
                            formal_statement=d["formal_statement"],
                            informal_statement=d.get("informal_statement"),
                            split=split,
                            source="miniF2F",
                        ))
            return problems

        # Try directory of .lean files
        split_dir = os.path.join(data_dir, split)
        if os.path.isdir(split_dir):
            for category in os.listdir(split_dir):
                cat_dir = os.path.join(split_dir, category)
                if not os.path.isdir(cat_dir):
                    continue
                for fname in os.listdir(cat_dir):
                    if not fname.endswith(".lean"):
                        continue
                    fpath = os.path.join(cat_dir, fname)
                    stmt = self._extract_theorem_statement(fpath)
                    if stmt:
                        problems.append(BenchmarkProblem(
                            name=fname[:-5],
                            formal_statement=stmt,
                            split=split,
                            category=category,
                            source="miniF2F",
                        ))
            return problems

        raise FileNotFoundError(
            f"MiniF2F data not found at {data_dir}. "
            "Download from https://github.com/openai/miniF2F"
        )

    def load_jsonl(
        self,
        path: str,
        name_field: str = "name",
        statement_field: str = "formal_statement",
        split_field: str = "split",
        target_split: Optional[str] = None,
        source: str = "unknown",
    ) -> List[BenchmarkProblem]:
        """
        Generic JSONL loader for LeanWorkbook, ProofNet, or custom benchmarks.

        Expected format (one JSON object per line):
            {"name": "...", "formal_statement": "theorem ... : ...", "split": "test"}
        """
        problems: List[BenchmarkProblem] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                split = d.get(split_field, "test")
                if target_split and split != target_split:
                    continue
                problems.append(BenchmarkProblem(
                    name=d.get(name_field, "unknown"),
                    formal_statement=d.get(statement_field, ""),
                    informal_statement=d.get("informal_statement") or d.get("informal"),
                    split=split,
                    category=d.get("category"),
                    source=source,
                ))
        return problems

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        problems: List[BenchmarkProblem],
        benchmark_name: str = "unknown",
        k_values: Optional[List[int]] = None,
        max_problems: Optional[int] = None,
        verbose: bool = True,
        live_log_path: Optional[str] = None,
    ) -> BenchmarkResults:
        """
        Evaluate on a list of BenchmarkProblems.

        Args:
            problems: Problems to evaluate.
            benchmark_name: Name for the result object.
            k_values: For Pass@k, attempt each problem k times with sampling.
            max_problems: If set, only evaluate the first N problems.
            verbose: Print progress.
        """
        if max_problems:
            problems = problems[:max_problems]

        t_start = time.time()
        solved = 0
        proof_times: List[float] = []
        tactic_counts: Dict[str, int] = {}
        solved_names: List[str] = []
        unsolved_names: List[str] = []
        solved_details: Dict[str, dict] = {}

        # Open live log for writing proven theorems as they come in
        _live_f = open(live_log_path, "a") if live_log_path else None

        for i, prob in enumerate(problems):
            if verbose:
                print(
                    f"[{benchmark_name}] {i+1}/{len(problems)}: {prob.name} ...",
                    end=" ", flush=True,
                )

            result = self.prover.prove(prob.formal_statement)

            if result.success:
                solved += 1
                proof_times.append(result.time_taken_seconds)
                solved_names.append(prob.name)
                tac = result.tactic_used or "unknown"
                tactic_counts[tac] = tactic_counts.get(tac, 0) + 1
                solved_details[prob.name] = {
                    "statement": prob.formal_statement,
                    "tactic": tac,
                    "proof_time_s": round(result.time_taken_seconds, 2),
                }
                if verbose:
                    print(f"✓ [{tac}] {result.time_taken_seconds:.1f}s", flush=True)
                if _live_f:
                    _live_f.write(json.dumps({
                        "name": prob.name,
                        "statement": prob.formal_statement,
                        "tactic": tac,
                        "proof_time_s": round(result.time_taken_seconds, 2),
                        "informal": prob.informal_statement,
                    }) + "\n")
                    _live_f.flush()
            else:
                unsolved_names.append(prob.name)
                if verbose:
                    print("✗", flush=True)

        if _live_f:
            _live_f.close()

        # Pass@k (requires multiple attempts with sampling temperature > 0)
        pass_at_k: Dict[int, float] = {}
        if k_values:
            for k in k_values:
                pass_at_k[k] = self._estimate_pass_at_k(problems, k, verbose=False)

        eval_time = time.time() - t_start
        n = len(problems)
        return BenchmarkResults(
            benchmark_name=benchmark_name,
            total_problems=n,
            solved=solved,
            pass_at_1=solved / max(n, 1),
            pass_at_k=pass_at_k,
            avg_proof_time_seconds=sum(proof_times) / max(len(proof_times), 1),
            solved_by_tactic=tactic_counts,
            unsolved_names=unsolved_names,
            solved_names=solved_names,
            solved_details=solved_details,
            evaluation_time_seconds=eval_time,
        )

    def evaluate_discovery(
        self,
        budget_seconds: float = 3600.0,
        snapshot_interval: float = 300.0,
    ) -> BenchmarkResults:
        """
        Open-ended mode: generate novel Lean 4 conjectures and prove them.
        Analogous to Track A's run_for() but in Lean 4 context.
        """
        if self.generator is None:
            raise ValueError("A Lean4ConjectureGenerator is required for discovery mode.")

        t_start = time.time()
        solved = 0
        attempted = 0
        tactic_counts: Dict[str, int] = {}
        solved_names: List[str] = []
        proof_times: List[float] = []
        batch_size = 20

        while (time.time() - t_start) < budget_seconds:
            conjectures = self.generator.generate(batch_size)
            for i, thm_str in enumerate(conjectures):
                if not thm_str:
                    continue
                attempted += 1
                result = self.prover.prove(thm_str)
                if result.success:
                    solved += 1
                    name = f"discovery_{attempted}"
                    solved_names.append(name)
                    proof_times.append(result.time_taken_seconds)
                    tac = result.tactic_used or "unknown"
                    tactic_counts[tac] = tactic_counts.get(tac, 0) + 1

            elapsed = time.time() - t_start
            if elapsed % snapshot_interval < 1.0:
                rate = solved / max(attempted, 1)
                print(
                    f"[discovery] t={elapsed:.0f}s | attempted={attempted} | "
                    f"solved={solved} | rate={rate:.2%}",
                    flush=True,
                )

        return BenchmarkResults(
            benchmark_name="lean4_discovery",
            total_problems=attempted,
            solved=solved,
            pass_at_1=solved / max(attempted, 1),
            avg_proof_time_seconds=sum(proof_times) / max(len(proof_times), 1),
            solved_by_tactic=tactic_counts,
            solved_names=solved_names,
            evaluation_time_seconds=time.time() - t_start,
        )

    def save_results(self, results: BenchmarkResults, path: str):
        """Save results to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"Results saved to {path}", flush=True)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _extract_theorem_statement(self, lean_file: str) -> Optional[str]:
        """
        Extract a theorem statement from a .lean file.
        Strips the `:= by ...` proof if present.
        """
        try:
            with open(lean_file) as f:
                content = f.read()
            # Find the first `theorem` declaration
            idx = content.find("theorem ")
            if idx == -1:
                return None
            decl = content[idx:]
            # Remove the proof part
            for sep in [" := by", "\n:= by", ":= by", ":=\n"]:
                proof_idx = decl.find(sep)
                if proof_idx != -1:
                    decl = decl[:proof_idx].strip()
                    break
            return decl.strip()
        except Exception:
            return None

    def _estimate_pass_at_k(
        self,
        problems: List[BenchmarkProblem],
        k: int,
        verbose: bool = False,
    ) -> float:
        """
        Estimate Pass@k by attempting each problem k times with varied temperature.
        A problem is "solved at k" if any of the k attempts succeeds.
        """
        if k <= 1:
            return 0.0  # Already computed as Pass@1

        solved_at_k = 0
        original_portfolio = self.prover.portfolio

        for prob in problems:
            for _ in range(k):
                result = self.prover.prove(prob.formal_statement)
                if result.success:
                    solved_at_k += 1
                    break

        return solved_at_k / max(len(problems), 1)
