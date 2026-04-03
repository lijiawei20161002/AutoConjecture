#!/usr/bin/env python3
"""
Extended comparison: AutoConjecture vs STP vs GPT-4o vs Random.

Uses the extended prover (induction, unification-based rewriting) introduced
after the baseline experiment on 2026-04-02.

Usage:
    # Quick smoke-test (5 min per system, skip GPT-4o):
    python scripts/run_extended_comparison.py --budget 300 --skip-llm

    # Full run (20 min per system, all systems):
    python scripts/run_extended_comparison.py --budget 1200

    # Include GPT-4o (requires OPENAI_API_KEY in env):
    python scripts/run_extended_comparison.py --budget 600

    # Override prover iterations:
    python scripts/run_extended_comparison.py --max-iter 5000 --budget 600
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── config ─────────────────────────────────────────────────────────────────────

DEFAULT_CFG = dict(
    # Prover — extended settings (was 500/50 before)
    max_iterations    = 5000,
    max_depth         = 30,
    timeout_per_proof = 30.0,
    parallel_workers  = 4,

    # Generator
    min_complexity    = 6,
    max_complexity    = 20,
    batch_size        = 10,
    seed              = 42,
    device            = "cpu",
    var_names         = ["x", "y", "z", "w"],

    # Neural generator (used by STP + AutoConj)
    gen_d_model       = 256,
    gen_nhead         = 8,
    gen_num_layers    = 6,
    gen_lr            = 1e-4,
    gen_batch_size    = 32,
    gen_warmup_steps  = 500,
    neural_ratio      = 0.5,
    update_interval   = 100,
    pretrain_epochs   = 0,

    # STP specific
    frontier_k        = 20,
    reinforce_lr      = 1e-5,
    reinforce_interval= 10,
    supervised_interval= 50,

    # GPT-4o specific
    llm_model         = "gpt-4o",
    llm_batch_size    = 10,
    llm_temperature   = 1.0,
    llm_max_tokens    = 800,

    # Diversity / novelty
    max_similar       = 20,
)


# ── runner builders ────────────────────────────────────────────────────────────

def build_runners(include_llm: bool, include_autoconj: bool):
    from src.baselines.random_baseline    import RandomBaselineRunner
    from src.baselines.heuristic_baseline import HeuristicBaselineRunner
    from src.baselines.stp_baseline       import STPBaselineRunner

    runners = [
        RandomBaselineRunner(),
        HeuristicBaselineRunner(),
        STPBaselineRunner(),
    ]

    if include_autoconj:
        from src.baselines.autoconj_baseline import AutoConjBaselineRunner
        runners.append(AutoConjBaselineRunner())

    if include_llm:
        try:
            from src.baselines.llm_baseline import LLMBaselineRunner
            runners.append(LLMBaselineRunner())
        except ImportError as e:
            print(f"WARNING: could not import LLMBaselineRunner: {e}", flush=True)

    return runners


# ── run loop ───────────────────────────────────────────────────────────────────

def run_all(runners, cfg, budget_s, snapshot_interval_s, out_dir):
    results = {}
    for runner in runners:
        print(f"\n{'='*70}", flush=True)
        print(f"  System: {runner.name}   budget={budget_s}s", flush=True)
        print(f"{'='*70}", flush=True)

        # Merge global config with any runner-specific overrides
        run_cfg = dict(cfg)
        # GPT-4o gets its own live log
        if runner.name == "llm_gpt4o":
            run_cfg["live_log_path"] = os.path.join(out_dir, "llm_live.jsonl")

        try:
            runner.setup(run_cfg)
        except Exception as e:
            print(f"  [{runner.name}] setup failed: {e}", flush=True)
            continue

        t0 = time.time()
        try:
            snapshots = runner.run_for(budget_s, snapshot_interval_s)
        except Exception as e:
            print(f"  [{runner.name}] run failed: {e}", flush=True)
            snapshots = []

        elapsed = time.time() - t0
        results[runner.name] = snapshots

        if snapshots:
            last = snapshots[-1]
            print(f"\n  [{runner.name}] FINAL: {last.summary_line()}", flush=True)
        print(f"  [{runner.name}] Wall-clock: {elapsed:.1f}s", flush=True)

        # Save immediately
        snap_path = os.path.join(out_dir, f"snapshots_{runner.name}.jsonl")
        with open(snap_path, "w") as f:
            for s in snapshots:
                f.write(json.dumps(s.to_dict()) + "\n")

        try:
            kb = runner.get_final_kb()
            thms = kb.get_all_theorems()
            kb_path = os.path.join(out_dir, f"kb_{runner.name}.jsonl")
            with open(kb_path, "w") as f:
                for t in thms:
                    f.write(json.dumps(t.to_dict()) + "\n")
            print(f"  [{runner.name}] {len(thms)} theorems saved to {kb_path}", flush=True)
        except Exception as e:
            print(f"  [{runner.name}] kb save failed: {e}", flush=True)

    return results


# ── reporting ──────────────────────────────────────────────────────────────────

def print_table(results):
    print("\n" + "=" * 100)
    print("COMPARISON TABLE  (extended prover: induction + unification rewriting)")
    print("=" * 100)

    hdr = (f"{'System':<20} {'Theorems':>8} {'Thm/hr':>8} {'Succ%':>7} "
           f"{'Non-triv%':>10} {'Div':>6} {'P@10':>6} {'P@20':>6}")
    print(hdr)
    print("-" * 100)

    for name, snaps in results.items():
        if not snaps:
            print(f"  {name:<20}  (no data)")
            continue
        s = snaps[-1]
        print(
            f"  {s.system_name:<20} "
            f"{s.theorems_unique:>8d} "
            f"{s.theorems_per_hour:>8.1f} "
            f"{s.proof_success_rate:>7.2%} "
            f"{s.non_triviality_rate:>10.2%} "
            f"{s.diversity_score:>6.3f} "
            f"{s.provability_at_k.get(10, 0):>6.2%} "
            f"{s.provability_at_k.get(20, 0):>6.2%} "
        )
    print("=" * 100)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extended comparison: all systems")
    parser.add_argument("--budget",      type=float, default=300.0,
                        help="Wall-clock budget per system in seconds (default 300)")
    parser.add_argument("--snapshot",    type=float, default=30.0,
                        help="Snapshot interval in seconds (default 30)")
    parser.add_argument("--max-iter",    type=int, default=None,
                        help="Override prover max_iterations")
    parser.add_argument("--skip-llm",    action="store_true",
                        help="Skip GPT-4o (no API key needed)")
    parser.add_argument("--skip-autoconj", action="store_true",
                        help="Skip full AutoConj Phase5 (saves time)")
    parser.add_argument("--systems",     type=str, default=None,
                        help="Comma-separated list of systems to run (e.g. llm_gpt4o,heuristic)")
    parser.add_argument("--output-dir",  type=str, default=None)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "data", "experiments", f"extended_{timestamp}"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput: {out_dir}")

    cfg = dict(DEFAULT_CFG)
    if args.max_iter:
        cfg["max_iterations"] = args.max_iter

    # Save run metadata
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "timestamp": timestamp,
            "budget_s": args.budget,
            "max_iterations": cfg["max_iterations"],
            "max_depth": cfg["max_depth"],
            "prover": "extended (induction + unification rewriting)",
            "skip_llm": args.skip_llm,
        }, f, indent=2)

    runners = build_runners(
        include_llm=not args.skip_llm,
        include_autoconj=not args.skip_autoconj,
    )

    if args.systems:
        keep = set(args.systems.split(","))
        runners = [r for r in runners if r.name in keep]
    print(f"Systems: {[r.name for r in runners]}")
    print(f"Budget:  {args.budget}s per system")
    print(f"Prover:  max_iterations={cfg['max_iterations']}, max_depth={cfg['max_depth']}")

    results = run_all(runners, cfg, args.budget, args.snapshot, out_dir)
    print_table(results)

    # Save combined summary JSON
    summary = {
        name: [s.to_dict() for s in snaps]
        for name, snaps in results.items()
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
