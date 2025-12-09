#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GQ-CNN Grasp Planner ROS2 Node

This node provides a ROS2 service for grasp planning using the PyTorch
implementation of GQ-CNN.

Dependencies:
    - gqcnn_torch (this package)
    - gqcnn_interfaces (ROS2 message/service definitions)
    - rclpy
    - sensor_msgs
    - geometry_msgs
"""
import numpy as np
import yaml
import os

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose

# Import interfaces (will be available after building gqcnn_interfaces)
try:
    from gqcnn_interfaces.srv import (
        GQCNNGraspPlanner,
        GQCNNGraspPlannerBoundingBox,
        GQCNNGraspPlannerSegmask,
        GQCNNGraspPlannerFull,
    )
    from gqcnn_interfaces.msg import GQCNNGrasp, BoundingBox
except ImportError:
    print("Warning: gqcnn_interfaces not found. Build the package first.")
    GQCNNGraspPlanner = None

# Import gqcnn_torch
from gqcnn_torch import get_gqcnn_model
from gqcnn_torch.grasping import RobustGraspingPolicy, RgbdImageState


class GraspPlannerNode(Node):
    """ROS2 node for GQ-CNN grasp planning."""

    def __init__(self):
        super().__init__("grasp_planner_node")

        # Declare parameters
        self.declare_parameter("model_path", "")
        self.declare_parameter("config_path", "")
        self.declare_parameter("gripper_width", 0.05)

        # Get parameters
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        config_path = self.get_parameter("config_path").get_parameter_value().string_value
        self.gripper_width = self.get_parameter("gripper_width").get_parameter_value().double_value

        # CV Bridge
        self.bridge = CvBridge()

        # Load model
        if model_path:
            self.get_logger().info(f"Loading GQ-CNN model from {model_path}")
            GQCNN = get_gqcnn_model(backend="torch")
            self.gqcnn = GQCNN.load(model_path)
            self.gqcnn.to_device()
            self.get_logger().info("Model loaded successfully")
        else:
            self.get_logger().warn("No model path specified. Services will not work.")
            self.gqcnn = None

        # Load config
        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            self.policy_config = config.get("policy", {})
        else:
            self.policy_config = self._default_policy_config()

        # Create policy
        if self.gqcnn is not None:
            self.policy = RobustGraspingPolicy(self.policy_config, self.gqcnn)
            self.get_logger().info("Policy created successfully")
        else:
            self.policy = None

        # Create services
        if GQCNNGraspPlanner is not None:
            self.grasp_srv = self.create_service(
                GQCNNGraspPlanner,
                "gqcnn/grasp_planner",
                self.grasp_planner_callback,
            )
            self.grasp_bbox_srv = self.create_service(
                GQCNNGraspPlannerBoundingBox,
                "gqcnn/grasp_planner_bounding_box",
                self.grasp_planner_bbox_callback,
            )
            self.grasp_segmask_srv = self.create_service(
                GQCNNGraspPlannerSegmask,
                "gqcnn/grasp_planner_segmask",
                self.grasp_planner_segmask_callback,
            )
            self.grasp_full_srv = self.create_service(
                GQCNNGraspPlannerFull,
                "gqcnn/grasp_planner_full",
                self.grasp_planner_full_callback,
            )
            self.get_logger().info("Services created successfully")
        else:
            self.get_logger().warn("gqcnn_interfaces not available. Services not created.")

        self.get_logger().info("Grasp planner node initialized")

    def _default_policy_config(self):
        """Return default policy configuration."""
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
                "max_dist_from_center": 160,
                "min_dist_from_boundary": 45,
                "min_grasp_dist": 2.5,
            },
            "metric": {
                "crop_height": 32,
                "crop_width": 32,
            },
        }

    def _depth_image_to_array(self, depth_msg):
        """Convert ROS Image message to numpy array."""
        depth_im = self.bridge.imgmsg_to_cv2(depth_msg)
        # Convert to meters if in mm
        if depth_im.dtype == np.uint16:
            depth_im = depth_im.astype(np.float32) / 1000.0
        return depth_im

    def _create_grasp_response(self, action, color_msg=None):
        """Create GQCNNGrasp message from action."""
        grasp_msg = GQCNNGrasp()

        # Pose
        grasp_msg.pose = Pose()
        grasp_msg.pose.position.x = float(action.grasp.center[0])
        grasp_msg.pose.position.y = float(action.grasp.center[1])
        grasp_msg.pose.position.z = float(action.grasp.depth)

        # Quaternion from angle (rotation around z-axis)
        angle = action.grasp.angle
        grasp_msg.pose.orientation.x = 0.0
        grasp_msg.pose.orientation.y = 0.0
        grasp_msg.pose.orientation.z = float(np.sin(angle / 2))
        grasp_msg.pose.orientation.w = float(np.cos(angle / 2))

        # Q-value
        grasp_msg.q_value = float(action.q_value)

        # Grasp type
        grasp_msg.grasp_type = GQCNNGrasp.PARALLEL_JAW

        # Center in pixels
        grasp_msg.center_px = [float(action.grasp.center[0]), float(action.grasp.center[1])]

        # Angle and depth
        grasp_msg.angle = float(action.grasp.angle)
        grasp_msg.depth = float(action.grasp.depth)

        return grasp_msg

    def _plan_grasp(self, depth_msg, camera_info_msg, color_msg=None, segmask_msg=None, bbox=None):
        """Plan a grasp from depth image."""
        if self.policy is None:
            self.get_logger().error("Policy not initialized")
            return None

        # Convert depth image
        depth_im = self._depth_image_to_array(depth_msg)
        if depth_im.ndim == 2:
            depth_im = depth_im[:, :, np.newaxis]

        # Create RGBD image
        h, w = depth_im.shape[:2]
        if color_msg is not None:
            color_im = self.bridge.imgmsg_to_cv2(color_msg, "rgb8")
        else:
            color_im = np.zeros((h, w, 3), dtype=np.uint8)

        rgbd_im = np.concatenate([color_im, depth_im], axis=-1)

        # Process segmask
        segmask = None
        if segmask_msg is not None:
            segmask = self.bridge.imgmsg_to_cv2(segmask_msg)

        # Apply bounding box to create mask if provided
        if bbox is not None and segmask is None:
            segmask = np.zeros((h, w), dtype=np.uint8)
            x1, y1 = int(bbox.min_x), int(bbox.min_y)
            x2, y2 = int(bbox.max_x), int(bbox.max_y)
            segmask[y1:y2, x1:x2] = 255

        # Create state
        state = RgbdImageState(rgbd_im, camera_intr=None, segmask=segmask)

        # Plan grasp
        try:
            action = self.policy.action(state)
            return action
        except Exception as e:
            self.get_logger().error(f"Grasp planning failed: {e}")
            return None

    def grasp_planner_callback(self, request, response):
        """Handle basic grasp planning request."""
        self.get_logger().info("Received grasp planning request")

        action = self._plan_grasp(
            request.depth_image,
            request.camera_info,
            request.color_image,
        )

        if action is not None:
            response.grasp = self._create_grasp_response(action, request.color_image)
            self.get_logger().info(f"Planned grasp with Q-value: {action.q_value:.4f}")
        else:
            self.get_logger().warn("Failed to plan grasp")

        return response

    def grasp_planner_bbox_callback(self, request, response):
        """Handle grasp planning with bounding box request."""
        self.get_logger().info("Received grasp planning request with bounding box")

        action = self._plan_grasp(
            request.depth_image,
            request.camera_info,
            request.color_image,
            bbox=request.bounding_box,
        )

        if action is not None:
            response.grasp = self._create_grasp_response(action)
            self.get_logger().info(f"Planned grasp with Q-value: {action.q_value:.4f}")

        return response

    def grasp_planner_segmask_callback(self, request, response):
        """Handle grasp planning with segmask request."""
        self.get_logger().info("Received grasp planning request with segmask")

        action = self._plan_grasp(
            request.depth_image,
            request.camera_info,
            request.color_image,
            segmask_msg=request.segmask,
        )

        if action is not None:
            response.grasp = self._create_grasp_response(action)
            self.get_logger().info(f"Planned grasp with Q-value: {action.q_value:.4f}")

        return response

    def grasp_planner_full_callback(self, request, response):
        """Handle full grasp planning request."""
        self.get_logger().info("Received full grasp planning request")

        action = self._plan_grasp(
            request.depth_image,
            request.camera_info,
            request.color_image,
            segmask_msg=request.segmask,
            bbox=request.bounding_box,
        )

        if action is not None:
            response.grasp = self._create_grasp_response(action)
            self.get_logger().info(f"Planned grasp with Q-value: {action.q_value:.4f}")

        return response


def main(args=None):
    rclpy.init(args=args)
    node = GraspPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
