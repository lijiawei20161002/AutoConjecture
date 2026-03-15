#!/usr/bin/env python3
"""
Phase 3 training script: RL-based prover with PPO.

Usage:
  python3 scripts/train_phase3.py [options]

Key options:
  --device cuda          GPU device (default: cuda if available)
  --epochs N             Number of training epochs (default: 20)
  --cycles N             Cycles per epoch (default: 500)
  --kb-checkpoint PATH   Phase 2 KB file for warm-start
  --experiment-name NAME Output prefix (default: phase3_a100)
  --bc-warmup N          Behavioral cloning warm-up cycles (default: 200)
  --no-heuristic         Disable heuristic fallback
"""
import argparse
import os
import sys
import time
import datetime

# Ensure project root is on Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.training.phase3_training_loop import Phase3TrainingLoop, Phase3Config
from src.monitoring.logger import Logger


def parse_args():
    p = argparse.ArgumentParser(
        description="AutoConjecture Phase 3: RL-based prover training"
    )

    # Training schedule
    p.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    p.add_argument("--cycles", type=int, default=500, help="Cycles per epoch")
    p.add_argument("--conjectures-per-cycle", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    # Generation
    p.add_argument("--neural-ratio", type=float, default=0.5,
                   help="Fraction of neural vs random generation (0–1)")
    p.add_argument("--initial-complexity", type=int, default=6)
    p.add_argument("--final-complexity", type=int, default=15)

    # RL prover
    p.add_argument("--max-proof-steps", type=int, default=30)
    p.add_argument("--update-interval", type=int, default=50,
                   help="PPO update every N conjectures")
    p.add_argument("--bc-warmup", type=int, default=200,
                   help="Behavioral cloning warm-up cycles")
    p.add_argument("--no-heuristic", action="store_true",
                   help="Disable heuristic fallback prover")

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

    # Device and paths
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--experiment-name", type=str, default="phase3_a100")
    p.add_argument("--checkpoint-dir", type=str, default="data/checkpoints")
    p.add_argument("--checkpoint-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--kb-checkpoint", type=str, default=None,
                   help="Path to Phase 2 KB JSON for warm-start")

    return p.parse_args()


def main():
    args = parse_args()

    # Set up experiment directory
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("data", "experiments", f"{args.experiment_name}_{ts}")
    os.makedirs(exp_dir, exist_ok=True)
    log_path = os.path.join(exp_dir, f"{args.experiment_name}_{ts}.log")

    logger = Logger(
        log_dir=exp_dir,
        experiment_name=f"{args.experiment_name}_{ts}",
    )

    logger.log("=" * 70)
    logger.log("AutoConjecture Phase 3: RL-Based Prover")
    logger.log(f"Started: {datetime.datetime.now().isoformat()}")
    logger.log(f"Device: {args.device}")
    if torch.cuda.is_available():
        logger.log(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.log(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    logger.log("=" * 70)

    # Auto-detect Phase 2 KB checkpoint if not specified
    kb_checkpoint = args.kb_checkpoint
    if kb_checkpoint is None:
        candidate = "data/checkpoints/neural_epoch_19_cycle_999.json"
        if os.path.exists(candidate):
            kb_checkpoint = candidate
            logger.log(f"Auto-detected Phase 2 KB: {candidate}")

    # Build config
    config = Phase3Config(
        num_epochs=args.epochs,
        cycles_per_epoch=args.cycles,
        conjectures_per_cycle=args.conjectures_per_cycle,
        random_seed=args.seed,
        neural_ratio=args.neural_ratio,
        initial_complexity=args.initial_complexity,
        final_complexity=args.final_complexity,
        max_proof_steps=args.max_proof_steps,
        update_interval=args.update_interval,
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
        device=args.device,
        experiment_name=args.experiment_name,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        kb_checkpoint=kb_checkpoint,
    )

    # Log config
    logger.log("\nConfiguration:")
    for k, v in config.__dict__.items():
        logger.log(f"  {k}: {v}")
    logger.log("")

    # Run training
    loop = Phase3TrainingLoop(config=config, logger=logger)

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
        logger.log(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
