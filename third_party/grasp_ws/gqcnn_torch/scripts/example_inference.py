#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example: Running GQ-CNN inference on example depth images.

This script demonstrates how to use the gqcnn_torch package to:
1. Load a trained GQ-CNN model
2. Run grasp planning on depth images
3. Visualize the results
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add package to path if not installed
pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from gqcnn_torch import get_gqcnn_model
from gqcnn_torch.grasping import RobustGraspingPolicy, RgbdImageState


def create_example_config():
    """Create a simple policy configuration."""
    return {
        "num_seed_samples": 128,
        "num_gmm_samples": 64,
        "num_iters": 3,
        "gmm_refit_p": 0.25,
        "gmm_component_frac": 0.4,
        "gmm_reg_covar": 0.01,
        "sampling": {
            "type": "antipodal_depth",
            "friction_coef": 1.0,
            "depth_grad_thresh": 0.0025,
            "depth_grad_gaussian_sigma": 1.0,
            "downsample_rate": 4,
            "max_rejection_samples": 4000,
            "max_dist_from_center": 160,
            "min_dist_from_boundary": 45,
            "min_grasp_dist": 2.5,
            "angle_dist_weight": 5.0,
            "depth_sampling_mode": "uniform",
            "depth_samples_per_grasp": 3,
            "min_depth_offset": 0.015,
            "max_depth_offset": 0.05,
        },
        "metric": {
            "crop_height": 32,
            "crop_width": 32,
        },
    }


def load_example_depth_image(data_dir):
    """Load an example depth image from the data directory."""
    depth_path = os.path.join(data_dir, "single_object", "primesense", "depth_0.npy")
    if os.path.exists(depth_path):
        return np.load(depth_path)
    else:
        print(f"Warning: Example depth image not found at {depth_path}")
        print("Generating synthetic depth image...")
        # Generate a synthetic depth image with an object
        h, w = 480, 640
        depth = np.ones((h, w), dtype=np.float32) * 0.8  # Background at 0.8m
        
        # Add a circular object
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        radius = 80
        mask = (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2
        depth[mask] = 0.5  # Object at 0.5m
        
        return depth


def visualize_grasp(depth_im, action):
    """Visualize the planned grasp on the depth image."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(depth_im, cmap="viridis")
    ax.set_title(f"Planned Grasp (Q-value: {action.q_value:.4f})")

    # Draw grasp
    center = action.grasp.center
    angle = action.grasp.angle
    length = 40

    dx = length * np.cos(angle)
    dy = length * np.sin(angle)

    # Draw grasp axis
    ax.plot(
        [center[0] - dx, center[0] + dx],
        [center[1] - dy, center[1] + dy],
        "r-",
        linewidth=3,
        label="Grasp axis",
    )

    # Draw perpendicular (gripper jaws)
    perp_dx = 20 * np.cos(angle + np.pi / 2)
    perp_dy = 20 * np.sin(angle + np.pi / 2)
    ax.plot(
        [center[0] + dx - perp_dx, center[0] + dx + perp_dx],
        [center[1] + dy - perp_dy, center[1] + dy + perp_dy],
        "g-",
        linewidth=2,
    )
    ax.plot(
        [center[0] - dx - perp_dx, center[0] - dx + perp_dx],
        [center[1] - dy - perp_dy, center[1] - dy + perp_dy],
        "g-",
        linewidth=2,
    )

    # Draw center point
    ax.plot(center[0], center[1], "ro", markersize=8, label="Grasp center")

    ax.legend()
    plt.colorbar(ax.images[0], ax=ax, label="Depth (m)")
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("GQ-CNN PyTorch Example - Grasp Planning")
    print("=" * 60)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(pkg_dir, "data", "examples")

    # Load example depth image
    print("\nLoading example depth image...")
    depth_im = load_example_depth_image(data_dir)
    print(f"  Depth image shape: {depth_im.shape}")
    print(f"  Depth range: [{depth_im.min():.3f}, {depth_im.max():.3f}] m")

    # Format for processing [H, W, 1]
    if depth_im.ndim == 2:
        depth_im_proc = depth_im[:, :, np.newaxis]
    else:
        depth_im_proc = depth_im

    # Create RGBD image (dummy color)
    h, w = depth_im.shape[:2]
    color_im = np.zeros((h, w, 3), dtype=np.uint8)
    rgbd_im = np.concatenate([color_im, depth_im_proc], axis=-1)

    # Create state
    state = RgbdImageState(rgbd_im, camera_intr=None, segmask=None)

    # Create policy configuration
    policy_config = create_example_config()

    # Note: For actual inference, you would load a trained model:
    # GQCNN = get_gqcnn_model(backend="torch")
    # gqcnn = GQCNN.load("path/to/trained/model")
    # policy = RobustGraspingPolicy(policy_config, gqcnn)

    # For this example, we'll demonstrate grasp sampling without a model
    print("\nDemonstrating grasp sampling (without model evaluation)...")
    from gqcnn_torch.grasping import ImageGraspSamplerFactory

    sampler = ImageGraspSamplerFactory.sampler(
        "antipodal_depth",
        policy_config["sampling"],
    )

    # Sample grasps
    grasps = sampler.sample(rgbd_im, None, num_samples=50)
    print(f"  Sampled {len(grasps)} grasp candidates")

    if len(grasps) > 0:
        # Just pick the first grasp for visualization
        from gqcnn_torch.grasping import GraspAction
        
        # Assign random quality for demonstration
        best_grasp = grasps[0]
        action = GraspAction(best_grasp, q_value=np.random.uniform(0.5, 1.0))

        print(f"\n  Example grasp:")
        print(f"    Center: ({action.grasp.center[0]:.1f}, {action.grasp.center[1]:.1f})")
        print(f"    Angle: {np.degrees(action.grasp.angle):.1f}°")
        print(f"    Depth: {action.grasp.depth:.3f} m")

        # Visualize
        print("\nVisualizing grasp...")
        visualize_grasp(depth_im, action)
    else:
        print("  No valid grasps found!")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
