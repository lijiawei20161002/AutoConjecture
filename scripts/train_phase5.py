#!/usr/bin/env python3
"""
Phase 5 training script: Optimization (parallel proving + advanced curriculum).

Usage:
  python3 scripts/train_phase5.py [options]

Key options:
  --device cuda            GPU device (default: cuda if available, else cpu)
  --epochs N               Training epochs (default: 20)
  --cycles N               Cycles per epoch (default: 500)
  --parallel-workers N     Parallel CPU prover processes (default: 4)
  --curriculum STRATEGY    self_paced | adaptive_band | linear (default: self_paced)
  --kb-checkpoint PATH     Phase 3/4 KB file for warm-start
  --experiment-name NAME   Output prefix (default: phase5_opt)
  --use-torch-compile      Enable torch.compile() (PyTorch >= 2.0)
"""
import argparse
import datetime
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.training.phase5_training_loop import Phase5TrainingLoop, Phase5Config
from src.monitoring.logger import Logger


def parse_args():
    p = argparse.ArgumentParser(
        description="AutoConjecture Phase 5: Optimization Training"
    )

    # Training schedule
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--cycles", type=int, default=500)
    p.add_argument("--conjectures-per-cycle", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    # Generation
    p.add_argument("--neural-ratio", type=float, default=0.5)
    p.add_argument("--initial-complexity", type=int, default=6)
    p.add_argument("--final-complexity", type=int, default=20)

    # Curriculum
    p.add_argument(
        "--curriculum",
        type=str,
        default="self_paced",
        choices=["self_paced", "adaptive_band", "linear"],
        help="Curriculum strategy",
    )
    p.add_argument("--spc-ema-alpha", type=float, default=0.1)
    p.add_argument("--spc-target-lo", type=float, default=0.10)
    p.add_argument("--spc-target-hi", type=float, default=0.60)
    p.add_argument("--spc-frontier-margin", type=int, default=3)
    p.add_argument("--abc-advance-threshold", type=float, default=0.40)
    p.add_argument("--abc-retreat-threshold", type=float, default=0.05)
    p.add_argument("--abc-patience", type=int, default=200)

    # Parallel prover
    p.add_argument(
        "--parallel-workers",
        type=int,
        default=4,
        help="CPU worker processes for heuristic prover (1 = sequential)",
    )
    p.add_argument("--heuristic-max-depth", type=int, default=50)
    p.add_argument("--heuristic-max-iter", type=int, default=500)
    p.add_argument("--heuristic-timeout", type=float, default=30.0)

    # RL prover
    p.add_argument("--max-proof-steps", type=int, default=30)
    p.add_argument("--ppo-update-interval", type=int, default=50)
    p.add_argument("--bc-warmup", type=int, default=200)
    p.add_argument("--no-heuristic", action="store_true")

    # State encoder
    p.add_argument("--enc-d-model", type=int, default=256)
    p.add_argument("--enc-layers", type=int, default=3)
    p.add_argument("--enc-heads", type=int, default=4)

    # PPO
    p.add_argument("--ppo-lr", type=float, default=3e-4)
    p.add_argument("--ppo-clip", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--ppo-batch", type=int, default=64)
    p.add_argument("--entropy-coef", type=float, default=0.01)

    # Generator
    p.add_argument("--gen-d-model", type=int, default=256)
    p.add_argument("--gen-layers", type=int, default=6)
    p.add_argument("--gen-pretrain-epochs", type=int, default=3)
    p.add_argument("--gen-update-interval", type=int, default=100)
    p.add_argument("--no-prioritized-buffer", action="store_true")

    # Optimization
    p.add_argument("--use-torch-compile", action="store_true")

    # Device and paths
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--experiment-name", type=str, default="phase5_opt")
    p.add_argument("--checkpoint-dir", type=str, default="data/checkpoints")
    p.add_argument("--checkpoint-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--kb-checkpoint", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("data", "experiments", f"{args.experiment_name}_{ts}")
    os.makedirs(exp_dir, exist_ok=True)

    logger = Logger(
        log_dir=exp_dir,
        experiment_name=f"{args.experiment_name}_{ts}",
    )

    logger.log("=" * 70)
    logger.log("AutoConjecture Phase 5: Optimization")
    logger.log(f"Started: {datetime.datetime.now().isoformat()}")
    logger.log(f"Device: {args.device}")
    if torch.cuda.is_available():
        logger.log(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.log(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    logger.log(f"Parallel workers: {args.parallel_workers}")
    logger.log(f"Curriculum: {args.curriculum}")
    logger.log("=" * 70)

    # Auto-detect KB checkpoint
    kb_checkpoint = args.kb_checkpoint
    if kb_checkpoint is None:
        for candidate in [
            "data/checkpoints/phase3_a100_fixed_kb_epoch_19_cycle_499.json",
            "data/checkpoints/neural_epoch_19_cycle_999.json",
            "data/checkpoints/neural_epoch_19_cycle_1000_reconstructed.json",
        ]:
            if os.path.exists(candidate):
                kb_checkpoint = candidate
                logger.log(f"Auto-detected KB: {candidate}")
                break

    config = Phase5Config(
        num_epochs=args.epochs,
        cycles_per_epoch=args.cycles,
        conjectures_per_cycle=args.conjectures_per_cycle,
        random_seed=args.seed,
        neural_ratio=args.neural_ratio,
        initial_complexity=args.initial_complexity,
        final_complexity=args.final_complexity,
        curriculum_strategy=args.curriculum,
        spc_ema_alpha=args.spc_ema_alpha,
        spc_target_lo=args.spc_target_lo,
        spc_target_hi=args.spc_target_hi,
        spc_frontier_margin=args.spc_frontier_margin,
        abc_advance_threshold=args.abc_advance_threshold,
        abc_retreat_threshold=args.abc_retreat_threshold,
        abc_patience=args.abc_patience,
        parallel_workers=args.parallel_workers,
        heuristic_max_depth=args.heuristic_max_depth,
        heuristic_max_iter=args.heuristic_max_iter,
        heuristic_timeout=args.heuristic_timeout,
        max_proof_steps=args.max_proof_steps,
        ppo_update_interval=args.ppo_update_interval,
        bc_warmup_cycles=args.bc_warmup,
        use_heuristic_fallback=not args.no_heuristic,
        encoder_d_model=args.enc_d_model,
        encoder_nhead=args.enc_heads,
        encoder_num_layers=args.enc_layers,
        ac_hidden_dim=args.enc_d_model,
        ppo_lr=args.ppo_lr,
        ppo_clip_epsilon=args.ppo_clip,
        ppo_epochs=args.ppo_epochs,
        ppo_mini_batch_size=args.ppo_batch,
        ppo_entropy_coef=args.entropy_coef,
        gen_d_model=args.gen_d_model,
        gen_num_layers=args.gen_layers,
        gen_pretrain_epochs=args.gen_pretrain_epochs,
        gen_update_interval=args.gen_update_interval,
        gen_use_prioritized_buffer=not args.no_prioritized_buffer,
        use_torch_compile=args.use_torch_compile,
        device=args.device,
        experiment_name=args.experiment_name,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        kb_checkpoint=kb_checkpoint,
    )

    logger.log("\nConfiguration:")
    for k, v in config.__dict__.items():
        logger.log(f"  {k}: {v}")
    logger.log("")

    loop = Phase5TrainingLoop(config=config, logger=logger)
    t0 = time.time()
    try:
        loop.train()
    except KeyboardInterrupt:
        logger.log("\n[Interrupted by user]")
    finally:
        elapsed = time.time() - t0
        stats = loop.get_statistics()
        logger.log(f"\nFinal statistics after {elapsed:.1f}s:")
        for k, v in stats.items():
            logger.log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
