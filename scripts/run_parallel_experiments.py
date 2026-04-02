#!/usr/bin/env python3
"""
Parallel experiment runner: LLM baseline + Lean 4 MiniF2F evaluation.

Launches in parallel:
  A) LLM (GPT-4o) conjecture generation — 3 seeds × 1 hr
  B) Lean 4 MiniF2F evaluation       — 3 seeds × 244 problems

Every proven theorem is written immediately to:
  data/experiments/parallel_<timestamp>/live_theorems.jsonl

Final JSON summaries per run go to:
  data/experiments/parallel_<timestamp>/<system>_seed<N>_results.json

Usage:
    python scripts/run_parallel_experiments.py
    python scripts/run_parallel_experiments.py --llm-budget 3600 --seeds 1 2 3
    python scripts/run_parallel_experiments.py --skip-lean   # LLM only
    python scripts/run_parallel_experiments.py --skip-llm    # MiniF2F only
    python scripts/run_parallel_experiments.py --minif2f-dir /path/to/miniF2F
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Worker functions (run in child processes) ──────────────────────────────────

def _worker_llm(seed: int, budget_s: float, out_dir: str, live_log: str) -> None:
    """LLM baseline worker for one seed."""
    import sys
    sys.path.insert(0, PROJECT_ROOT)

    from src.baselines.llm_baseline import LLMBaselineRunner

    tag = f"llm_gpt4o_seed{seed}"
    run_live = os.path.join(out_dir, f"{tag}_live.jsonl")

    runner = LLMBaselineRunner()
    runner.setup({
        "seed": seed,
        "llm_model": "gpt-4o",
        "llm_batch_size": 10,
        "llm_temperature": 0.9 + seed * 0.05,   # slight variation across seeds
        "live_log_path": run_live,
        "max_depth": 50,
        "max_iterations": 500,
    })

    print(f"[{tag}] Starting {budget_s}s run...", flush=True)
    snapshots = runner.run_for(
        wall_clock_budget_seconds=budget_s,
        snapshot_interval_seconds=60.0,
    )

    # Save snapshots
    out_path = os.path.join(out_dir, f"{tag}_snapshots.json")
    with open(out_path, "w") as f:
        json.dump([s.to_dict() for s in snapshots], f, indent=2)

    # Save final KB
    kb = runner.get_final_kb()
    kb_path = os.path.join(out_dir, f"{tag}_kb.json")
    _save_kb(kb, kb_path)

    # Append proven theorems to the shared live log
    _merge_live_log(run_live, live_log, tag)

    final = snapshots[-1]
    print(
        f"[{tag}] DONE — theorems={final.theorems_unique} "
        f"success_rate={final.proof_success_rate:.3f}",
        flush=True,
    )


def _worker_minif2f(seed: int, data_path: str, out_dir: str, live_log: str,
                    lean_exec: str, no_mathlib: bool) -> None:
    """MiniF2F evaluation worker for one seed."""
    import sys
    sys.path.insert(0, PROJECT_ROOT)

    # Import only the REPL and prover directly to avoid torch dependency
    import importlib.util

    def _load_module(rel_path):
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        spec = importlib.util.spec_from_file_location(rel_path.replace("/", "."), abs_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "src.lean4"
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    repl_mod = _load_module("src/lean4/repl_interface.py")
    prover_mod = _load_module("src/lean4/lean4_prover.py")
    evaluator_mod = _load_module("src/lean4/benchmark_evaluator.py")

    Lean4REPLInterface = repl_mod.Lean4REPLInterface
    Lean4TacticProver = prover_mod.Lean4TacticProver
    BenchmarkEvaluator = evaluator_mod.BenchmarkEvaluator
    BenchmarkProblem = evaluator_mod.BenchmarkProblem

    tag = f"minif2f_seed{seed}"

    repl = Lean4REPLInterface(
        lean_executable=lean_exec,
        timeout_seconds=30.0,
        mode="oneshot",
        use_mathlib=not no_mathlib,
    )
    if not repl.is_available():
        print(f"[{tag}] Lean 4 not available at '{lean_exec}'. Skipping.", flush=True)
        _write_error(out_dir, tag, f"Lean 4 not found at {lean_exec}")
        return

    live_path = os.path.join(out_dir, f"{tag}_live.jsonl")

    with repl:
        portfolio = (prover_mod.NO_MATHLIB_TACTIC_PORTFOLIO if no_mathlib
                     else prover_mod.MATHLIB_TACTIC_PORTFOLIO)
        prover = Lean4TacticProver(repl, tactic_portfolio=portfolio, timeout_per_tactic=8.0)
        evaluator = BenchmarkEvaluator(repl, prover)

        # Load problems — support both JSONL file and directory
        if os.path.isfile(data_path) and data_path.endswith(".jsonl"):
            print(f"[{tag}] Loading MiniF2F JSONL from {data_path}...", flush=True)
            try:
                problems = evaluator.load_jsonl(
                    data_path, target_split="test", source="miniF2F"
                )
            except Exception as e:
                print(f"[{tag}] JSONL load failed: {e}", flush=True)
                _write_error(out_dir, tag, str(e))
                return
        elif os.path.isdir(data_path):
            print(f"[{tag}] Loading MiniF2F directory from {data_path}...", flush=True)
            try:
                problems = evaluator.load_minif2f(data_path, split="test")
            except FileNotFoundError as e:
                print(f"[{tag}] {e}", flush=True)
                _write_error(out_dir, tag, str(e))
                return
        else:
            print(f"[{tag}] data path not found: {data_path}", flush=True)
            _write_error(out_dir, tag, f"data not found: {data_path}")
            return

        # When running without Mathlib, skip statements that use types/lemmas
        # requiring Mathlib (ℝ, ℤ, ℂ, Prime, Finset, etc.).  These fail
        # immediately but each lean invocation still costs ~3s of startup time.
        if no_mathlib:
            _MATHLIB_TOKENS = {"ℝ", "ℤ", "ℂ", "ℚ", "Prime", "Finset", "Matrix",
                               "Complex", "Real", "Int.", "Rat.", "Polynomial",
                               "Multiset", "Set.", "Filter.", "MvPolynomial"}
            before = len(problems)
            problems = [p for p in problems
                        if not any(t in p.formal_statement for t in _MATHLIB_TOKENS)]
            print(f"[{tag}] After Nat-only filter: {len(problems)}/{before} problems", flush=True)

        print(f"[{tag}] Evaluating {len(problems)} problems...", flush=True)
        results = evaluator.evaluate(
            problems,
            benchmark_name=f"miniF2F-test-seed{seed}",
            verbose=True,
            live_log_path=live_path,
        )

        # Save results
        out_path = os.path.join(out_dir, f"{tag}_results.json")
        evaluator.save_results(results, out_path)

        # Append proven theorems to shared live log
        _merge_live_log(live_path, live_log, tag)

        print(
            f"[{tag}] DONE — solved={results.solved}/{results.total_problems} "
            f"pass@1={results.pass_at_1:.3f}",
            flush=True,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save_kb(kb, path: str) -> None:
    """Serialize a KnowledgeBase to JSON."""
    try:
        data = {
            "size": len(kb.get_all_statements()),
            "theorems": [str(t) for t in kb.get_all_statements()],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save KB to {path}: {e}", flush=True)


def _merge_live_log(src: str, dst: str, tag: str) -> None:
    """Copy entries from a per-seed live log to the shared log, adding a tag."""
    if not os.path.isfile(src):
        return
    with open(src) as sf, open(dst, "a") as df:
        for line in sf:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entry["worker"] = tag
                df.write(json.dumps(entry) + "\n")
            except Exception:
                pass


def _write_error(out_dir: str, tag: str, msg: str) -> None:
    path = os.path.join(out_dir, f"{tag}_error.txt")
    with open(path, "w") as f:
        f.write(msg + "\n")


def _find_lean(lean_path_hint: Optional[str]) -> str:
    """Return path to lean executable, checking common install locations."""
    candidates = [lean_path_hint] if lean_path_hint else []
    candidates += [
        os.path.expanduser("~/.elan/bin/lean"),   # elan install (most common)
        "lean",
        "/usr/local/bin/lean",
        "/opt/homebrew/bin/lean",
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
        try:
            import subprocess
            r = subprocess.run([c, "--version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return c
        except Exception:
            pass
    return "lean"  # fall back; worker will report not-available


def _print_summary(out_dir: str) -> None:
    """Print a combined summary table from all result files."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    # LLM snapshots
    for f in sorted(Path(out_dir).glob("llm_*_snapshots.json")):
        try:
            snaps = json.loads(f.read_text())
            last = snaps[-1] if snaps else {}
            tag = f.stem.replace("_snapshots", "")
            print(
                f"  {tag:30s}  theorems={last.get('theorems',0):4d} "
                f"success_rate={last.get('success_rate',0):.3f}"
            )
        except Exception:
            pass

    # MiniF2F results
    for f in sorted(Path(out_dir).glob("minif2f_*_results.json")):
        try:
            r = json.loads(f.read_text())
            tag = f.stem.replace("_results", "")
            print(
                f"  {tag:30s}  solved={r.get('solved',0):4d}/{r.get('total',0)} "
                f"pass@1={r.get('pass_at_1',0):.3f}"
            )
        except Exception:
            pass

    # Shared live log stats
    live_log = os.path.join(out_dir, "live_theorems.jsonl")
    if os.path.isfile(live_log):
        with open(live_log) as f:
            entries = [json.loads(l) for l in f if l.strip()]
        print(f"\n  Total proven theorems logged: {len(entries)}")
        by_worker: dict = {}
        for e in entries:
            w = e.get("worker", "unknown")
            by_worker[w] = by_worker.get(w, 0) + 1
        for w, cnt in sorted(by_worker.items()):
            print(f"    {w}: {cnt}")

    print("=" * 70)
    print(f"All results in: {out_dir}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel LLM + MiniF2F experiment runner")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--llm-budget", type=float, default=3600.0,
                        help="Wall-clock budget per LLM seed (seconds, default 3600)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM baseline")
    parser.add_argument("--skip-lean", action="store_true", help="Skip MiniF2F eval")
    default_minif2f = os.path.join(PROJECT_ROOT, "data", "benchmarks", "minif2f_lean4.jsonl")
    parser.add_argument("--minif2f-dir", type=str, default=default_minif2f,
                        help="Path to miniF2F JSONL (or lean4 directory)")
    parser.add_argument("--lean-exec", type=str, default=None,
                        help="Path to lean executable (default: auto-detect)")
    parser.add_argument("--no-mathlib", action="store_true",
                        help="Run Lean 4 without Mathlib (faster, fewer tactics)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "data", "experiments", f"parallel_{timestamp}"
    )
    os.makedirs(out_dir, exist_ok=True)

    live_log = os.path.join(out_dir, "live_theorems.jsonl")
    lean_exec = _find_lean(args.lean_exec)

    # Write experiment metadata
    meta = {
        "timestamp": timestamp,
        "seeds": args.seeds,
        "llm_budget_s": args.llm_budget,
        "skip_llm": args.skip_llm,
        "skip_lean": args.skip_lean,
        "minif2f_dir": args.minif2f_dir,
        "lean_exec": lean_exec,
        "no_mathlib": args.no_mathlib,
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nOutput directory: {out_dir}")
    print(f"Live theorem log: {live_log}")
    print(f"Seeds: {args.seeds}")
    print(f"Lean executable: {lean_exec}")
    print()

    # ── spawn workers ──────────────────────────────────────────────────────────
    procs: List[mp.Process] = []

    if not args.skip_llm:
        for seed in args.seeds:
            p = mp.Process(
                target=_worker_llm,
                args=(seed, args.llm_budget, out_dir, live_log),
                name=f"llm_seed{seed}",
                daemon=False,
            )
            p.start()
            procs.append(p)
            print(f"Started LLM worker seed={seed} PID={p.pid}", flush=True)

    if not args.skip_lean:
        for seed in args.seeds:
            p = mp.Process(
                target=_worker_minif2f,
                args=(seed, args.minif2f_dir, out_dir, live_log,
                      lean_exec, args.no_mathlib),
                name=f"minif2f_seed{seed}",
                daemon=False,
            )
            p.start()
            procs.append(p)
            print(f"Started MiniF2F worker seed={seed} PID={p.pid}", flush=True)

    if not procs:
        print("No workers started. Use --skip-llm / --skip-lean to filter.")
        return 1

    # ── monitor loop ───────────────────────────────────────────────────────────
    print(f"\nAll {len(procs)} workers running. Monitoring every 60s...\n")

    try:
        while any(p.is_alive() for p in procs):
            time.sleep(60)
            alive = [p.name for p in procs if p.is_alive()]
            done = [p.name for p in procs if not p.is_alive()]

            # Count live theorems so far
            n_live = 0
            if os.path.isfile(live_log):
                with open(live_log) as f:
                    n_live = sum(1 for l in f if l.strip())

            print(
                f"[monitor] alive={alive}  done={done}  "
                f"live_theorems={n_live}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nInterrupted — terminating workers...", flush=True)
        for p in procs:
            if p.is_alive():
                p.terminate()

    # ── wait for all to finish ─────────────────────────────────────────────────
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            print(f"Warning: worker {p.name} did not exit cleanly", flush=True)
            p.kill()

    _print_summary(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
