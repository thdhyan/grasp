#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train GQ-CNN model.
Example script for training GQ-CNN models using PyTorch.
"""
import argparse
import logging
import os
import yaml

from gqcnn_torch import get_gqcnn_trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train GQ-CNN model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to training data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/gqcnn_torch",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Override data directory
    config["data_dir"] = args.data_dir

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get trainer class
    GQCNNTrainer = get_gqcnn_trainer(backend="torch")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = GQCNNTrainer(config, output_dir=args.output_dir)

    # Resume from checkpoint if specified
    if args.resume is not None:
        logger.info(f"Resuming from {args.resume}")
        trainer._gqcnn.load_weights(args.resume)

    # Train
    logger.info("Starting training...")
    trainer.train()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
