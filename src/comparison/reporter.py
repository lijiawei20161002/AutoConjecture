"""
Report generation for comparison results.

Produces:
  - Markdown comparison table (for the paper)
  - CSV for further analysis
  - PNG learning-curve plots
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

from .metrics import ComparisonSnapshot


class ComparisonReporter:
    """Generates comparison reports from snapshot collections."""

    def to_markdown_table(
        self,
        results: Dict[str, List[ComparisonSnapshot]],
        at_seconds: Optional[float] = None,
    ) -> str:
        """
        Emit a paper-ready Markdown comparison table.

        If at_seconds is given, uses the snapshot closest to that time.
        Otherwise uses the final snapshot for each system.
        """
        rows = []
        for name, snaps in results.items():
            if not snaps:
                continue
            snap = self._pick_snapshot(snaps, at_seconds)
            p10 = snap.provability_at_k.get(10, 0.0)
            p20 = snap.provability_at_k.get(20, 0.0)
            rows.append((
                snap.system_name,
                snap.theorems_unique,
                snap.theorems_per_hour,
                snap.proof_success_rate,
                snap.non_triviality_rate,
                snap.avg_complexity,
                snap.diversity_score,
                p10,
                p20,
            ))

        # Sort by theorems discovered descending
        rows.sort(key=lambda r: r[1], reverse=True)

        header = (
            "| System | Thm | Thm/hr | Succ% | Non-triv% | Avg-cplx | Diversity | P@10 | P@20 |"
        )
        sep = (
            "|--------|-----|--------|-------|-----------|----------|-----------|------|------|"
        )
        lines = [header, sep]
        for (sys, thm, tph, sr, nt, ac, div, p10, p20) in rows:
            lines.append(
                f"| {sys:<16} "
                f"| {thm:5d} "
                f"| {tph:7.1f} "
                f"| {sr:6.2%} "
                f"| {nt:9.2%} "
                f"| {ac:8.1f} "
                f"| {div:9.3f} "
                f"| {p10:5.2%} "
                f"| {p20:5.2%} |"
            )
        return "\n".join(lines)

    def to_csv(
        self,
        results: Dict[str, List[ComparisonSnapshot]],
        path: str,
        include_all_snapshots: bool = True,
    ):
        """Save all snapshots (or just final ones) to CSV."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        fieldnames = [
            "system", "wall_clock_s", "theorems", "attempted", "proved",
            "theorems_per_hour", "success_rate", "avg_complexity",
            "non_triviality_rate", "diversity_score",
            "p_at_5", "p_at_10", "p_at_20", "p_at_50",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, snaps in results.items():
                iter_snaps = snaps if include_all_snapshots else snaps[-1:]
                for snap in iter_snaps:
                    writer.writerow({
                        "system": snap.system_name,
                        "wall_clock_s": round(snap.wall_clock_seconds, 2),
                        "theorems": snap.theorems_unique,
                        "attempted": snap.attempted,
                        "proved": snap.proved,
                        "theorems_per_hour": round(snap.theorems_per_hour, 2),
                        "success_rate": round(snap.proof_success_rate, 4),
                        "avg_complexity": round(snap.avg_complexity, 2),
                        "non_triviality_rate": round(snap.non_triviality_rate, 4),
                        "diversity_score": round(snap.diversity_score, 4),
                        "p_at_5": round(snap.provability_at_k.get(5, 0.0), 4),
                        "p_at_10": round(snap.provability_at_k.get(10, 0.0), 4),
                        "p_at_20": round(snap.provability_at_k.get(20, 0.0), 4),
                        "p_at_50": round(snap.provability_at_k.get(50, 0.0), 4),
                    })
        print(f"CSV saved to {path}", flush=True)

    def to_json(
        self,
        results: Dict[str, List[ComparisonSnapshot]],
        path: str,
    ):
        """Save full snapshot data to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            name: [s.to_dict() for s in snaps]
            for name, snaps in results.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"JSON saved to {path}", flush=True)

    def plot_curves(
        self,
        results: Dict[str, List[ComparisonSnapshot]],
        output_dir: str,
    ):
        """
        Emit PNG learning-curve plots.
        Requires matplotlib; silently skips if unavailable.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plots.", flush=True)
            return

        os.makedirs(output_dir, exist_ok=True)

        # ── Plot 1: Theorems over time ─────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, snaps in results.items():
            if not snaps:
                continue
            xs = [s.wall_clock_seconds / 60.0 for s in snaps]
            ys = [s.theorems_unique for s in snaps]
            ax.plot(xs, ys, marker="o", markersize=3, label=name)
        ax.set_xlabel("Wall-clock time (minutes)")
        ax.set_ylabel("Unique theorems discovered")
        ax.set_title("Theorem Discovery over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "theorems_over_time.png"), dpi=150)
        plt.close(fig)
        print(f"Saved theorems_over_time.png", flush=True)

        # ── Plot 2: Proof success rate over time ───────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, snaps in results.items():
            if not snaps:
                continue
            xs = [s.wall_clock_seconds / 60.0 for s in snaps]
            ys = [s.proof_success_rate * 100 for s in snaps]
            ax.plot(xs, ys, marker="o", markersize=3, label=name)
        ax.set_xlabel("Wall-clock time (minutes)")
        ax.set_ylabel("Proof success rate (%)")
        ax.set_title("Proof Success Rate over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "success_rate_over_time.png"), dpi=150)
        plt.close(fig)
        print(f"Saved success_rate_over_time.png", flush=True)

        # ── Plot 3: Non-triviality rate over time ──────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, snaps in results.items():
            if not snaps:
                continue
            xs = [s.wall_clock_seconds / 60.0 for s in snaps]
            ys = [s.non_triviality_rate * 100 for s in snaps]
            ax.plot(xs, ys, marker="o", markersize=3, label=name)
        ax.set_xlabel("Wall-clock time (minutes)")
        ax.set_ylabel("Non-triviality rate (%)")
        ax.set_title("Theorem Non-Triviality over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "non_triviality_over_time.png"), dpi=150)
        plt.close(fig)
        print(f"Saved non_triviality_over_time.png", flush=True)

        # ── Plot 4: Diversity score over time ──────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, snaps in results.items():
            if not snaps:
                continue
            xs = [s.wall_clock_seconds / 60.0 for s in snaps]
            ys = [s.diversity_score for s in snaps]
            ax.plot(xs, ys, marker="o", markersize=3, label=name)
        ax.set_xlabel("Wall-clock time (minutes)")
        ax.set_ylabel("Diversity score (normalised entropy)")
        ax.set_title("Theorem Diversity over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "diversity_over_time.png"), dpi=150)
        plt.close(fig)
        print(f"Saved diversity_over_time.png", flush=True)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _pick_snapshot(
        self,
        snaps: List[ComparisonSnapshot],
        at_seconds: Optional[float],
    ) -> ComparisonSnapshot:
        if at_seconds is None or not snaps:
            return snaps[-1]
        closest = min(snaps, key=lambda s: abs(s.wall_clock_seconds - at_seconds))
        return closest
