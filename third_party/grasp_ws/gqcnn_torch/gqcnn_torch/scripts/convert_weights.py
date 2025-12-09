#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert TensorFlow GQ-CNN weights to PyTorch.
"""
import argparse
import logging
import os

from gqcnn_torch.utils import convert_tf_to_pytorch, load_tf_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Convert TensorFlow GQ-CNN to PyTorch"
    )
    parser.add_argument(
        "--tf_model",
        type=str,
        required=True,
        help="Path to TensorFlow model directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save PyTorch model (default: tf_model/model.pt)",
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = os.path.join(args.tf_model, "model.pt")

    logger.info(f"Converting TensorFlow model from {args.tf_model}")
    logger.info(f"Output will be saved to {args.output}")

    # Convert weights
    try:
        convert_tf_to_pytorch(args.tf_model, args.output, verbose=True)
        logger.info("Conversion complete!")
    except ImportError as e:
        logger.error(
            "TensorFlow is required for weight conversion. "
            "Install with: pip install tensorflow"
        )
        raise e
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise e


if __name__ == "__main__":
    main()
