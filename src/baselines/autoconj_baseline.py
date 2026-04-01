"""
AutoConjecture Phase 5 wrapped as a BaselineRunner.

Allows ComparisonRunner to treat the full Phase 5 system on equal footing
with the other baselines (same budget, same snapshot protocol).
"""
from __future__ import annotations

import time
from typing import List

from ..training.phase5_training_loop import Phase5TrainingLoop, Phase5Config
from ..knowledge.knowledge_base import KnowledgeBase
from ..comparison.metrics import ComparisonMetrics, ComparisonSnapshot
from .base_runner import BaselineRunner


class AutoConjBaselineRunner(BaselineRunner):
    """
    Full AutoConjecture Phase 5 (RL + neural generator + curriculum).
    Wraps Phase5TrainingLoop to expose the BaselineRunner interface.
    """

    @property
    def name(self) -> str:
        return "autoconj_ph5"

    def setup(self, config: dict) -> None:
        # Build a Phase5Config from the flat comparison config dict
        p5_cfg = Phase5Config(
            num_epochs=config.get("num_epochs", 999),       # Effectively unlimited; we stop by wall-clock
            cycles_per_epoch=config.get("cycles_per_epoch", 10_000),
            conjectures_per_cycle=config.get("batch_size", 10),
            random_seed=config.get("seed", 42),
            initial_complexity=config.get("min_complexity", 6),
            final_complexity=config.get("max_complexity", 20),
            neural_ratio=config.get("neural_ratio", 0.5),
            curriculum_strategy=config.get("curriculum_strategy", "self_paced"),
            parallel_workers=config.get("parallel_workers", 4),
            heuristic_max_depth=config.get("max_depth", 50),
            heuristic_max_iter=config.get("max_iterations", 500),
            heuristic_timeout=config.get("timeout_per_proof", 30.0),
            max_proof_steps=config.get("max_proof_steps", 30),
            ppo_update_interval=config.get("ppo_update_interval", 50),
            bc_warmup_cycles=config.get("bc_warmup_cycles", 200),
            gen_d_model=config.get("gen_d_model", 256),
            gen_nhead=config.get("gen_nhead", 8),
            gen_num_layers=config.get("gen_num_layers", 6),
            gen_lr=config.get("gen_lr", 1e-4),
            gen_batch_size=config.get("gen_batch_size", 32),
            gen_warmup_steps=config.get("gen_warmup_steps", 500),
            gen_pretrain_epochs=config.get("pretrain_epochs", 0),
            gen_update_interval=config.get("update_interval", 100),
            device=config.get("device", "cpu"),
            checkpoint_dir=config.get("checkpoint_dir", "data/checkpoints"),
            experiment_name=config.get("experiment_name", "comparison_autoconj"),
            checkpoint_interval=config.get("checkpoint_interval", 99_999),
            log_interval=config.get("log_interval", 500),
            kb_checkpoint=config.get("kb_checkpoint", None),
        )
        self._loop = Phase5TrainingLoop(p5_cfg)
        self.metrics = ComparisonMetrics(self.name)
        self._snapshots: List[ComparisonSnapshot] = []

    def run_for(
        self,
        wall_clock_budget_seconds: float,
        snapshot_interval_seconds: float = 60.0,
    ) -> List[ComparisonSnapshot]:
        """
        Run Phase 5 training until the wall-clock budget is exhausted.
        Snapshots are emitted every snapshot_interval_seconds.
        """
        snapshots: List[ComparisonSnapshot] = []
        kb = self._loop.kb
        t_start = time.time()
        t_last_snap = t_start

        cfg = self._loop.config

        # Run epoch → cycle loop, checking wall-clock after each cycle
        for epoch in range(cfg.num_epochs):
            self._loop.current_epoch = epoch
            self._loop.novelty_scorer.reset()
            self._loop.diversity_filter.reset()

            for cycle in range(cfg.cycles_per_epoch):
                now = time.time()
                if (now - t_start) >= wall_clock_budget_seconds:
                    break

                self._loop.current_cycle = cycle
                self._loop._run_cycle()

                if cycle % cfg.gen_update_interval == 0 and cycle > 0:
                    self._loop._update_generator()

                # Snapshot check
                now = time.time()
                if (now - t_last_snap) >= snapshot_interval_seconds:
                    snap = self.metrics.snapshot(
                        kb,
                        now - t_start,
                        self._loop.total_attempted,
                        self._loop.total_proved_rl + self._loop.total_proved_heuristic,
                    )
                    snapshots.append(snap)
                    print(snap.summary_line(), flush=True)
                    t_last_snap = now

            else:
                # Inner loop completed; check outer wall-clock
                if (time.time() - t_start) >= wall_clock_budget_seconds:
                    break
                continue
            break  # Inner loop broke early

        final_snap = self.metrics.snapshot(
            kb,
            time.time() - t_start,
            self._loop.total_attempted,
            self._loop.total_proved_rl + self._loop.total_proved_heuristic,
        )
        snapshots.append(final_snap)
        return snapshots

    def get_final_kb(self) -> KnowledgeBase:
        return self._loop.kb
