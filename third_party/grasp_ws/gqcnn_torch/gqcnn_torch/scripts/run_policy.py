#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run GQ-CNN grasp planning policy.
Example script for running trained GQ-CNN models.
"""
import argparse
import logging
import os
import yaml

import numpy as np

from gqcnn_torch import get_gqcnn_model
from gqcnn_torch.grasping import RobustGraspingPolicy, RgbdImageState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_depth_image(depth_path):
    """Load depth image from file."""
    if depth_path.endswith(".npy"):
        depth = np.load(depth_path)
    else:
        import cv2
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32) / 1000.0  # Convert to meters
    return depth


def main():
    parser = argparse.ArgumentParser(description="Run GQ-CNN policy")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to policy configuration YAML file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained GQ-CNN model directory",
    )
    parser.add_argument(
        "--depth",
        type=str,
        required=True,
        help="Path to depth image file (.npy or image)",
    )
    parser.add_argument(
        "--segmask",
        type=str,
        default=None,
        help="Path to segmentation mask image (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output visualization",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display visualization",
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    policy_config = config.get("policy", {})

    # Load model
    logger.info(f"Loading GQ-CNN model from {args.model}")
    GQCNN = get_gqcnn_model(backend="torch")
    gqcnn = GQCNN.load(args.model)

    # Move to GPU if available
    gqcnn.to_device()

    # Create policy
    logger.info("Creating policy...")
    policy = RobustGraspingPolicy(policy_config, gqcnn)

    # Load depth image
    logger.info(f"Loading depth image from {args.depth}")
    depth_im = load_depth_image(args.depth)

    # Ensure correct shape [H, W, 1]
    if depth_im.ndim == 2:
        depth_im = depth_im[:, :, np.newaxis]

    # Create RGBD image (dummy color for now)
    h, w = depth_im.shape[:2]
    color_im = np.zeros((h, w, 3), dtype=np.uint8)
    rgbd_im = np.concatenate([color_im, depth_im], axis=-1)

    # Load segmask if provided
    segmask = None
    if args.segmask is not None:
        import cv2
        segmask = cv2.imread(args.segmask, cv2.IMREAD_GRAYSCALE)

    # Create state
    state = RgbdImageState(rgbd_im, camera_intr=None, segmask=segmask)

    # Plan grasp
    logger.info("Planning grasp...")
    try:
        action = policy.action(state)
        logger.info(f"Planned grasp: {action}")
        logger.info(f"  Center: {action.grasp.center}")
        logger.info(f"  Angle: {action.grasp.angle:.3f} rad")
        logger.info(f"  Depth: {action.grasp.depth:.3f} m")
        logger.info(f"  Q-value: {action.q_value:.4f}")
    except Exception as e:
        logger.error(f"Grasp planning failed: {e}")
        return

    # Visualize if requested
    if args.visualize or args.output:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(depth_im[:, :, 0], cmap="gray")

        # Draw grasp
        center = action.grasp.center
        angle = action.grasp.angle
        length = 30

        dx = length * np.cos(angle)
        dy = length * np.sin(angle)

        ax.plot(
            [center[0] - dx, center[0] + dx],
            [center[1] - dy, center[1] + dy],
            "r-",
            linewidth=3,
        )
        ax.plot(center[0], center[1], "ro", markersize=10)
        ax.set_title(f"Q-value: {action.q_value:.4f}")

        if args.output:
            plt.savefig(args.output)
            logger.info(f"Saved visualization to {args.output}")

        if args.visualize:
            plt.show()


if __name__ == "__main__":
    main()
