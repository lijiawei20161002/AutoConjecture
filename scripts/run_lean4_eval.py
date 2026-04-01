#!/usr/bin/env python3
"""
Track B: Lean 4 evaluation of AutoConjecture.

Evaluates the AutoConjecture Lean 4 integration on standard benchmarks
(MiniF2F, LeanWorkbook, ProofNet) and in open-ended discovery mode.

Prerequisites:
  1. Lean 4 installed (via elan): https://leanprover.github.io/lean4/doc/setup.html
  2. (Optional) Mathlib4 project for imports:
       lake new MyProject && cd MyProject && lake update
  3. (Optional) lean4-repl for persistent mode:
       https://github.com/leanprover-community/repl

Usage examples:
    # Verify Lean 4 is available
    python scripts/run_lean4_eval.py --check-only

    # Run open-ended discovery for 30 minutes
    python scripts/run_lean4_eval.py --mode discovery --budget 1800

    # Evaluate on MiniF2F (requires local data)
    python scripts/run_lean4_eval.py --mode minif2f --data-dir /path/to/miniF2F

    # Evaluate on a custom JSONL benchmark file
    python scripts/run_lean4_eval.py --mode jsonl --data-path /path/to/bench.jsonl

    # Use persistent REPL (faster batch evaluation)
    python scripts/run_lean4_eval.py --repl-mode persistent --mode discovery

    # Use a Lake project with Mathlib
    python scripts/run_lean4_eval.py --project-dir /path/to/my_project --mode minif2f
"""
import argparse
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="AutoConjecture Lean 4 evaluation (Track B)"
    )
    parser.add_argument(
        "--mode",
        choices=["discovery", "minif2f", "jsonl", "templates"],
        default="discovery",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only verify Lean 4 is available, then exit",
    )
    parser.add_argument(
        "--budget", type=float, default=1800,
        help="Wall-clock budget in seconds (discovery mode only)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to benchmark data directory (miniF2F mode)",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to JSONL benchmark file (jsonl mode)",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Benchmark split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--max-problems", type=int, default=None,
        help="Limit evaluation to first N problems (for quick tests)",
    )
    parser.add_argument(
        "--repl-mode", choices=["oneshot", "persistent"], default="oneshot",
        help="How to interact with Lean 4 (default: oneshot)",
    )
    parser.add_argument(
        "--lean-executable", type=str, default="lean",
        help="Path to lean executable",
    )
    parser.add_argument(
        "--lake-executable", type=str, default="lake",
        help="Path to lake executable",
    )
    parser.add_argument(
        "--project-dir", type=str, default=None,
        help="Lake project directory with Mathlib dependency",
    )
    parser.add_argument(
        "--no-mathlib", action="store_true",
        help="Disable Mathlib import (faster, fewer tactics)",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0,
        help="Per-theorem timeout in seconds",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/lean4_results",
        help="Directory to write results",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for generator",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from src.lean4.repl_interface import Lean4REPLInterface, Lean4NotAvailable
    from src.lean4.lean4_prover import Lean4TacticProver
    from src.lean4.lean4_generator import Lean4ConjectureGenerator
    from src.lean4.benchmark_evaluator import BenchmarkEvaluator

    # ── Check Lean 4 availability ──────────────────────────────────────────────
    repl = Lean4REPLInterface(
        lean_executable=args.lean_executable,
        lake_executable=args.lake_executable,
        project_dir=args.project_dir,
        timeout_seconds=args.timeout,
        mode=args.repl_mode,
        use_mathlib=not args.no_mathlib,
    )

    if not repl.is_available():
        print(
            "ERROR: Lean 4 is not available.\n"
            "Install via elan: https://leanprover.github.io/lean4/doc/setup.html\n"
            "  curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh\n"
            "  elan toolchain install leanprover/lean4:stable",
            flush=True,
        )
        return 1

    print(f"Lean 4 is available at: {args.lean_executable}", flush=True)

    if args.check_only:
        # Quick sanity test
        print("Running sanity check: 1 + 1 = 2 ...", flush=True)
        with repl:
            result = repl.check_theorem("theorem sanity : 1 + 1 = 2 := by norm_num")
            if result.success:
                print(f"  ✓ Lean 4 works correctly ({result.time_taken_seconds:.2f}s)", flush=True)
            else:
                print(f"  ✗ Lean 4 check failed: {result.error_message}", flush=True)
        return 0

    # ── Main evaluation ────────────────────────────────────────────────────────
    generator = Lean4ConjectureGenerator(seed=args.seed)

    with repl:
        prover = Lean4TacticProver(repl, timeout_per_tactic=args.timeout / 5)
        evaluator = BenchmarkEvaluator(repl, prover, generator)

        results = None

        if args.mode == "discovery":
            print(f"\nOpen-ended Lean 4 discovery ({args.budget}s budget)...", flush=True)
            results = evaluator.evaluate_discovery(budget_seconds=args.budget)

        elif args.mode == "minif2f":
            if not args.data_dir:
                print("ERROR: --data-dir required for miniF2F mode.", flush=True)
                return 1
            print(f"Loading MiniF2F from {args.data_dir} (split={args.split})...", flush=True)
            problems = evaluator.load_minif2f(args.data_dir, split=args.split)
            print(f"Loaded {len(problems)} problems.", flush=True)
            results = evaluator.evaluate(
                problems,
                benchmark_name=f"miniF2F-{args.split}",
                max_problems=args.max_problems,
                verbose=True,
            )

        elif args.mode == "jsonl":
            if not args.data_path:
                print("ERROR: --data-path required for jsonl mode.", flush=True)
                return 1
            print(f"Loading benchmark from {args.data_path}...", flush=True)
            bench_name = os.path.splitext(os.path.basename(args.data_path))[0]
            problems = evaluator.load_jsonl(
                args.data_path,
                target_split=args.split if args.split != "all" else None,
                source=bench_name,
            )
            print(f"Loaded {len(problems)} problems.", flush=True)
            results = evaluator.evaluate(
                problems,
                benchmark_name=bench_name,
                max_problems=args.max_problems,
                verbose=True,
            )

        elif args.mode == "templates":
            # Quick evaluation of all built-in Lean 4 templates
            from src.lean4.benchmark_evaluator import BenchmarkProblem
            templates = generator.all_template_theorems()
            problems = [
                BenchmarkProblem(
                    name=f"template_{i}",
                    formal_statement=t,
                    source="autoconj_templates",
                )
                for i, t in enumerate(templates)
            ]
            print(f"Evaluating {len(problems)} built-in Lean 4 templates...", flush=True)
            live_path = os.path.join(args.output_dir, "proven_templates_live.jsonl")
            results = evaluator.evaluate(
                problems,
                benchmark_name="autoconj_templates",
                verbose=True,
                live_log_path=live_path,
            )
            print(f"Proven theorems live log: {live_path}", flush=True)

        if results is not None:
            print("\n" + "=" * 60, flush=True)
            print(results.summary(), flush=True)
            print("=" * 60, flush=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(
                args.output_dir,
                f"{results.benchmark_name}_{timestamp}.json"
            )
            evaluator.save_results(results, out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
