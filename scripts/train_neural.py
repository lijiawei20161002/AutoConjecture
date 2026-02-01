#!/usr/bin/env python3
"""
Training script for Phase 2: Neural Conjecture Generator

This script trains the neural conjecture generator with curriculum learning.

Usage:
    python scripts/train_neural.py [--config path/to/config.yaml] [options]

Examples:
    # Basic training with defaults
    python scripts/train_neural.py

    # Use custom config
    python scripts/train_neural.py --config configs/neural_phase2.yaml

    # Override specific parameters
    python scripts/train_neural.py --epochs 20 --device cuda

    # Continue from checkpoint
    python scripts/train_neural.py --resume data/checkpoints/generator_epoch_5.pt
"""

import argparse
import sys
import os
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.neural_training_loop import NeuralTrainingLoop, NeuralTrainingConfig
from src.monitoring.logger import Logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train neural conjecture generator (Phase 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file"
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1000,
        help="Cycles per epoch (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # Model parameters
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Model dimension (default: 256)"
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers (default: 6)"
    )

    # Generation parameters
    parser.add_argument(
        "--conjectures-per-cycle",
        type=int,
        default=10,
        help="Conjectures to generate per cycle (default: 10)"
    )

    # Curriculum
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable curriculum learning"
    )
    parser.add_argument(
        "--initial-complexity",
        type=int,
        default=2,
        help="Initial complexity for curriculum (default: 2)"
    )
    parser.add_argument(
        "--final-complexity",
        type=int,
        default=15,
        help="Final complexity for curriculum (default: 15)"
    )

    # Training strategy
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=5,
        help="Epochs of supervised pretraining (default: 5)"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to train on (default: cpu)"
    )

    # Checkpointing
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N cycles (default: 1000)"
    )

    # Experiment naming
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Check for CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Create config
    config = NeuralTrainingConfig(
        num_epochs=args.epochs,
        cycles_per_epoch=args.cycles,
        random_seed=args.seed,
        conjectures_per_cycle=args.conjectures_per_cycle,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        use_curriculum=not args.no_curriculum,
        initial_complexity=args.initial_complexity,
        final_complexity=args.final_complexity,
        pretrain_epochs=args.pretrain_epochs,
        device=args.device,
        checkpoint_interval=args.checkpoint_interval
    )

    # Create experiment directory
    if args.experiment_name:
        exp_dir = Path(f"data/experiments/{args.experiment_name}")
        exp_dir.mkdir(parents=True, exist_ok=True)
        log_path = exp_dir / "training.log"
    else:
        log_path = "data/logs/neural_training.log"

    # Ensure directories exist
    Path("data/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = Logger(
        log_dir=str(Path(log_path).parent),
        experiment_name=args.experiment_name
    )

    # Print configuration
    logger.log("="*60)
    logger.log("AutoConjecture Phase 2: Neural Generator Training")
    logger.log("="*60)
    logger.log(f"Configuration:")
    logger.log(f"  Epochs: {config.num_epochs}")
    logger.log(f"  Cycles per epoch: {config.cycles_per_epoch}")
    logger.log(f"  Conjectures per cycle: {config.conjectures_per_cycle}")
    logger.log(f"  Model: d_model={config.d_model}, heads={config.nhead}, layers={config.num_layers}")
    logger.log(f"  Device: {config.device}")
    logger.log(f"  Curriculum: {'enabled' if config.use_curriculum else 'disabled'}")
    if config.use_curriculum:
        logger.log(f"    Complexity: {config.initial_complexity} -> {config.final_complexity}")
    logger.log(f"  Pretrain epochs: {config.pretrain_epochs}")
    logger.log(f"  Random seed: {config.random_seed}")
    logger.log("="*60 + "\n")

    # Create training loop
    training_loop = NeuralTrainingLoop(
        config=config,
        logger=logger
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.log(f"Resuming from checkpoint: {args.resume}")
        training_loop.generator_trainer.load_checkpoint(args.resume)

    # Train
    try:
        training_loop.train()
    except KeyboardInterrupt:
        logger.log("\nTraining interrupted by user")
        logger.log("Saving checkpoint...")
        training_loop._checkpoint()
        logger.log("Checkpoint saved")
        return

    # Print final statistics
    stats = training_loop.get_statistics()
    logger.log("\n" + "="*60)
    logger.log("Final Statistics:")
    logger.log("="*60)
    logger.log(f"Knowledge base size: {stats['knowledge_base_size']}")
    logger.log(f"Total proofs found: {stats['total_proofs_succeeded']}")
    logger.log(f"Success rate: {stats['success_rate']:.2%}")
    logger.log(f"Average complexity: {stats.get('avg_complexity', 'N/A')}")

    if config.use_curriculum and 'curriculum' in stats:
        curr_stats = stats['curriculum']
        logger.log(f"\nCurriculum Progress:")
        logger.log(f"  Final stage: {curr_stats['current_stage']}")
        logger.log(f"  Final complexity: {curr_stats['current_complexity']}")
        logger.log(f"  Completed: {curr_stats['is_complete']}")

    logger.log("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
