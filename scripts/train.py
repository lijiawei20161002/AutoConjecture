#!/usr/bin/env python3
"""
Main training script for AutoConjecture.
Runs the generate-prove-learn cycle.

Usage:
    python scripts/train.py [--config CONFIG_PATH]
"""
import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.training_loop import TrainingLoop, TrainingConfig
from src.monitoring.logger import Logger
from src.monitoring.metrics import MetricsTracker
from src.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train AutoConjecture system")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Cycles per epoch (overrides config)"
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    print("="*60)
    print("AutoConjecture - AI Mathematical Reasoning System")
    print("="*60)

    # Load configuration
    try:
        config_dict = load_config(args.config)
        print(f"\nLoaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Warning: Config file not found at {args.config}")
        print("Using default configuration")
        config_dict = {}

    # Create training config
    training_cfg = config_dict.get("training", {})
    config = TrainingConfig(
        num_epochs=args.epochs if args.epochs else training_cfg.get("num_epochs", 10),
        cycles_per_epoch=args.cycles if args.cycles else training_cfg.get("cycles_per_epoch", 1000),
        random_seed=training_cfg.get("random_seed", 42),
        min_complexity=training_cfg.get("min_complexity", 2),
        max_complexity=training_cfg.get("max_complexity", 10),
        conjectures_per_cycle=training_cfg.get("conjectures_per_cycle", 10),
        max_proof_depth=training_cfg.get("max_proof_depth", 50),
        max_proof_iterations=training_cfg.get("max_proof_iterations", 1000),
        log_interval=training_cfg.get("log_interval", 100),
        checkpoint_interval=training_cfg.get("checkpoint_interval", 1000),
    )

    print("\nTraining Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Cycles per epoch: {config.cycles_per_epoch}")
    print(f"  Complexity range: [{config.min_complexity}, {config.max_complexity}]")
    print(f"  Random seed: {config.random_seed}")

    # Initialize logger
    logger = Logger(
        log_dir="data/logs",
        experiment_name=args.experiment_name
    )

    # Initialize metrics tracker
    metrics = MetricsTracker(save_path="data/logs/metrics.json")

    # Create training loop
    training_loop = TrainingLoop(config=config, logger=logger)

    # Run training
    try:
        logger.section("Starting Training")
        training_loop.train()

        # Get final statistics
        stats = training_loop.get_statistics()
        logger.section("Final Statistics")
        logger.log(f"Total conjectures generated: {stats['total_conjectures_generated']}")
        logger.log(f"Total proofs attempted: {stats['total_proofs_attempted']}")
        logger.log(f"Total proofs succeeded: {stats['total_proofs_succeeded']}")
        logger.log(f"Success rate: {stats['success_rate']:.2%}")
        logger.log(f"Knowledge base size: {stats['knowledge_base_size']}")
        logger.log(f"Average proof complexity: {stats.get('avg_complexity', 0):.2f}")

        # Save metrics
        metrics.log_metrics(stats, step=config.num_epochs * config.cycles_per_epoch)
        metrics.save()
        metrics.print_summary()

        logger.section("Training Complete!")
        logger.log(f"Logs saved to: {logger.log_file}")
        logger.log(f"Checkpoints saved to: data/checkpoints/")

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        logger.log("Saving checkpoint before exit...")
        training_loop._checkpoint()
        metrics.save()

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
