"""
Orchestrates multiple baseline runners under identical conditions.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from ..baselines.base_runner import BaselineRunner
from .metrics import ComparisonSnapshot


class ComparisonRunner:
    """
    Runs N baseline systems sequentially under the same wall-clock budget
    and snapshot interval, then returns all results for reporting.
    """

    def __init__(
        self,
        runners: List[BaselineRunner],
        global_config: dict,
    ):
        """
        Args:
            runners: List of BaselineRunner instances to evaluate.
            global_config: Shared configuration applied to every runner.
                           Runner-specific overrides can live under
                           global_config[runner.name].
        """
        self.runners = runners
        self.global_config = global_config
        self.budget_seconds: float = global_config["budget_seconds"]
        self.snapshot_interval: float = global_config.get(
            "snapshot_interval_seconds", 60.0
        )

    def run_sequential(
        self,
        output_dir: Optional[str] = None,
    ) -> Dict[str, List[ComparisonSnapshot]]:
        """
        Run each system one after another.
        Safe on single-GPU machines (only one GPU process at a time).

        If output_dir is given, saves per-baseline results immediately after
        each system finishes (don't lose data if a later baseline crashes).
        """
        results: Dict[str, List[ComparisonSnapshot]] = {}

        for runner in self.runners:
            print(f"\n{'='*70}", flush=True)
            print(f"Running baseline: {runner.name}", flush=True)
            print(f"{'='*70}", flush=True)

            # Merge global config with runner-specific overrides
            cfg = dict(self.global_config)
            cfg.update(self.global_config.get(runner.name, {}))

            runner.setup(cfg)
            t0 = time.time()
            snapshots = runner.run_for(self.budget_seconds, self.snapshot_interval)
            elapsed = time.time() - t0

            results[runner.name] = snapshots
            if snapshots:
                print(f"\n[{runner.name}] Final: {snapshots[-1].summary_line()}", flush=True)
            print(f"[{runner.name}] Wall-clock: {elapsed:.1f}s", flush=True)

            # Save this baseline's results immediately
            if output_dir:
                self._save_baseline(runner, snapshots, output_dir)

        return results

    def _save_baseline(
        self,
        runner,
        snapshots: List[ComparisonSnapshot],
        output_dir: str,
    ):
        """Persist snapshots + full theorem list for one baseline."""
        os.makedirs(output_dir, exist_ok=True)
        name = runner.name

        # Snapshot metrics (time-series)
        snap_path = os.path.join(output_dir, f"snapshots_{name}.jsonl")
        with open(snap_path, "w") as f:
            for s in snapshots:
                f.write(json.dumps(s.to_dict()) + "\n")
        print(f"[{name}] Snapshots -> {snap_path}", flush=True)

        # Full theorem list from the KB
        try:
            kb = runner.get_final_kb()
            theorems = kb.get_all_theorems()
            thm_path = os.path.join(output_dir, f"theorems_{name}.jsonl")
            with open(thm_path, "w") as f:
                for thm in theorems:
                    f.write(json.dumps(thm.to_dict()) + "\n")
            print(f"[{name}] {len(theorems)} theorems -> {thm_path}", flush=True)
        except Exception as e:
            print(f"[{name}] WARNING: could not save theorems: {e}", flush=True)

    def save_results(
        self,
        results: Dict[str, List[ComparisonSnapshot]],
        output_path: str,
    ):
        """Persist raw snapshot data to JSON for later analysis."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        data = {
            name: [s.to_dict() for s in snaps]
            for name, snaps in results.items()
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {output_path}", flush=True)
