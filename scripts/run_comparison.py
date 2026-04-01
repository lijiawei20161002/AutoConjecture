#!/usr/bin/env python3
"""
Track A: Run baseline comparison within the Peano arithmetic framework.

Usage examples:
    # Run all 5 systems for 1 hour each (uses comparison.yaml defaults)
    python scripts/run_comparison.py

    # Quick smoke-test: 5 minutes per system
    python scripts/run_comparison.py --budget 300

    # Run only specific baselines
    python scripts/run_comparison.py --baselines random heuristic supervised

    # Start from an existing KB checkpoint (loads theorems into each system)
    python scripts/run_comparison.py --kb-checkpoint data/checkpoints/phase3_kb.json

    # Use GPU for neural baselines
    python scripts/run_comparison.py --device cuda

    # Custom output directory
    python scripts/run_comparison.py --output-dir data/comparison_results/run01
"""
import argparse
import json
import os
import sys
import time

# ── Project root on path ───────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_yaml_config(path: str) -> dict:
    """Load YAML config with graceful fallback if PyYAML is absent."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        print("PyYAML not installed; using default config values.", flush=True)
        return {}
    except FileNotFoundError:
        print(f"Config file {path} not found; using defaults.", flush=True)
        return {}


def build_config(args) -> dict:
    """Merge YAML config with CLI overrides."""
    config_path = os.path.join(PROJECT_ROOT, "configs", "comparison.yaml")
    cfg = load_yaml_config(config_path)

    # CLI overrides
    if args.budget:
        cfg["budget_seconds"] = args.budget
    if args.device:
        cfg["device"] = args.device
    if args.kb_checkpoint:
        cfg["kb_checkpoint"] = args.kb_checkpoint
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.snapshot_interval:
        cfg["snapshot_interval_seconds"] = args.snapshot_interval

    # Ensure required keys have defaults
    cfg.setdefault("budget_seconds", 3600)
    cfg.setdefault("snapshot_interval_seconds", 60)
    cfg.setdefault("max_depth", 50)
    cfg.setdefault("max_iterations", 500)
    cfg.setdefault("min_complexity", 6)
    cfg.setdefault("max_complexity", 20)
    cfg.setdefault("batch_size", 10)
    cfg.setdefault("seed", 42)
    cfg.setdefault("device", "cpu")
    cfg.setdefault("gen_d_model", 256)
    cfg.setdefault("gen_nhead", 8)
    cfg.setdefault("gen_num_layers", 6)
    cfg.setdefault("gen_lr", 1e-4)
    cfg.setdefault("gen_batch_size", 32)
    cfg.setdefault("gen_warmup_steps", 500)
    cfg.setdefault("neural_ratio", 0.5)
    cfg.setdefault("parallel_workers", 4)
    cfg.setdefault("pretrain_epochs", 0)
    cfg.setdefault("update_interval", 100)

    return cfg


def build_runners(baseline_names):
    """Instantiate the requested baseline runners."""
    from src.baselines.random_baseline import RandomBaselineRunner
    from src.baselines.heuristic_baseline import HeuristicBaselineRunner
    from src.baselines.supervised_baseline import SupervisedBaselineRunner
    from src.baselines.stp_baseline import STPBaselineRunner
    from src.baselines.autoconj_baseline import AutoConjBaselineRunner

    available = {
        "random": RandomBaselineRunner,
        "heuristic": HeuristicBaselineRunner,
        "supervised": SupervisedBaselineRunner,
        "stp": STPBaselineRunner,
        "autoconj_ph5": AutoConjBaselineRunner,
    }

    if baseline_names is None:
        baseline_names = list(available.keys())

    runners = []
    for name in baseline_names:
        if name not in available:
            print(f"WARNING: Unknown baseline '{name}', skipping.", flush=True)
            continue
        runners.append(available[name]())
    return runners


def main():
    parser = argparse.ArgumentParser(
        description="AutoConjecture baseline comparison (Track A)"
    )
    parser.add_argument(
        "--baselines", nargs="+",
        choices=["random", "heuristic", "supervised", "stp", "autoconj_ph5"],
        default=None,
        help="Which baselines to run (default: all five)",
    )
    parser.add_argument(
        "--budget", type=float, default=None,
        help="Wall-clock budget in seconds per system (overrides comparison.yaml)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for neural baselines: 'cpu' or 'cuda'",
    )
    parser.add_argument(
        "--kb-checkpoint", type=str, default=None,
        help="Path to KB JSON checkpoint to seed each system with existing theorems",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--snapshot-interval", type=float, default=None,
        help="Seconds between metrics snapshots (default: 60)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/comparison_results",
        help="Directory to write results JSON, CSV, and plots",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip matplotlib plot generation",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build config ───────────────────────────────────────────────────────────
    cfg = build_config(args)
    print("\nComparison configuration:", flush=True)
    print(f"  Budget per system : {cfg['budget_seconds']}s", flush=True)
    print(f"  Device            : {cfg['device']}", flush=True)
    print(f"  Max depth         : {cfg['max_depth']}", flush=True)
    print(f"  Max iterations    : {cfg['max_iterations']}", flush=True)
    print(f"  Batch size        : {cfg['batch_size']}", flush=True)
    print(f"  Output dir        : {args.output_dir}", flush=True)

    # ── Build runners ──────────────────────────────────────────────────────────
    runners = build_runners(args.baselines)
    print(f"\nRunning {len(runners)} baseline(s): {[r.name for r in runners]}", flush=True)

    # ── Run comparison ─────────────────────────────────────────────────────────
    from src.comparison.runner import ComparisonRunner
    from src.comparison.reporter import ComparisonReporter

    runner = ComparisonRunner(runners, cfg)
    t0 = time.time()
    results = runner.run_sequential(output_dir=args.output_dir)
    total_time = time.time() - t0

    # ── Save results ───────────────────────────────────────────────────────────
    reporter = ComparisonReporter()
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(args.output_dir, f"results_{timestamp}.json")
    csv_path = os.path.join(args.output_dir, f"results_{timestamp}.csv")
    table_path = os.path.join(args.output_dir, f"table_{timestamp}.md")
    plots_dir = os.path.join(args.output_dir, f"plots_{timestamp}")

    reporter.to_json(results, json_path)
    reporter.to_csv(results, csv_path)

    md_table = reporter.to_markdown_table(results)
    with open(table_path, "w") as f:
        f.write("# AutoConjecture Baseline Comparison\n\n")
        f.write(md_table)
        f.write("\n")
    print(f"\nMarkdown table saved to {table_path}", flush=True)

    if not args.no_plots:
        reporter.plot_curves(results, plots_dir)

    # ── Print final table ──────────────────────────────────────────────────────
    print("\n" + "=" * 90, flush=True)
    print("FINAL COMPARISON TABLE", flush=True)
    print("=" * 90, flush=True)
    print(md_table, flush=True)
    print(f"\nTotal wall-clock time: {total_time:.1f}s", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
