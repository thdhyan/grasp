#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch file for GQ-CNN grasp planner node.

Usage:
    ros2 launch gqcnn_torch grasp_planner.launch.py \
        model_path:=/path/to/model \
        config_path:=/path/to/config.yaml
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare arguments
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="",
        description="Path to trained GQ-CNN model directory"
    )

    config_path_arg = DeclareLaunchArgument(
        "config_path",
        default_value="",
        description="Path to policy configuration YAML file"
    )

    gripper_width_arg = DeclareLaunchArgument(
        "gripper_width",
        default_value="0.05",
        description="Gripper width in meters"
    )

    # Create node
    grasp_planner_node = Node(
        package="gqcnn_torch",
        executable="grasp_planner_node.py",
        name="grasp_planner",
        parameters=[{
            "model_path": LaunchConfiguration("model_path"),
            "config_path": LaunchConfiguration("config_path"),
            "gripper_width": LaunchConfiguration("gripper_width"),
        }],
        output="screen",
    )

    return LaunchDescription([
        model_path_arg,
        config_path_arg,
        gripper_width_arg,
        grasp_planner_node,
    ])
