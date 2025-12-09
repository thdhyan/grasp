# -*- coding: utf-8 -*-
"""
GQ-CNN network implemented in PyTorch.
Based on the TensorFlow implementation from Berkeley Autolab.
"""
from collections import OrderedDict
import errno
from functools import reduce
import json
import math
import operator
import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import (
    reduce_shape,
    read_pose_data,
    pose_dim,
    weight_name_to_layer_name,
    GripperMode,
    TrainingMode,
    InputDepthMode,
    GQCNNFilenames,
)

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Convolutional block with ReLU, optional normalization, and pooling."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        pool_size,
        pool_stride,
        relu_coeff=0.0,
        norm=False,
        padding="same",
        normalization_params=None,
    ):
        super().__init__()
        self.relu_coeff = relu_coeff
        self.norm = norm
        self.normalization_params = normalization_params or {}

        # Calculate padding
        if padding.lower() == "same":
            pad = kernel_size // 2
        else:
            pad = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad)
        self.pool = nn.MaxPool2d(pool_size, stride=pool_stride, padding=pool_size // 2)

    def forward(self, x):
        x = self.conv(x)
        # Leaky ReLU
        x = F.leaky_relu(x, negative_slope=self.relu_coeff)

        if self.norm:
            # Local response normalization
            x = F.local_response_norm(
                x,
                size=self.normalization_params.get("radius", 2) * 2 + 1,
                alpha=self.normalization_params.get("alpha", 1e-4),
                beta=self.normalization_params.get("beta", 0.75),
                k=self.normalization_params.get("bias", 1.0),
            )

        x = self.pool(x)
        return x


class FCBlock(nn.Module):
    """Fully connected block with optional ReLU and dropout."""

    def __init__(self, in_features, out_features, relu_coeff=0.0, final_layer=False):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu_coeff = relu_coeff
        self.final_layer = final_layer

    def forward(self, x, drop_rate=0.0):
        x = self.fc(x)
        if not self.final_layer:
            x = F.leaky_relu(x, negative_slope=self.relu_coeff)
        if drop_rate > 0:
            x = F.dropout(x, p=drop_rate, training=self.training)
        return x


class PoseBlock(nn.Module):
    """Pose processing block."""

    def __init__(self, in_features, out_features, relu_coeff=0.0):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu_coeff = relu_coeff

    def forward(self, x):
        x = self.fc(x)
        x = F.leaky_relu(x, negative_slope=self.relu_coeff)
        return x


class MergeBlock(nn.Module):
    """Merge block for combining image and pose streams."""

    def __init__(self, in_features_im, in_features_pose, out_features, relu_coeff=0.0):
        super().__init__()
        self.fc_im = nn.Linear(in_features_im, out_features)
        self.fc_pose = nn.Linear(in_features_pose, out_features)
        self.relu_coeff = relu_coeff

    def forward(self, x_im, x_pose, drop_rate=0.0):
        x = self.fc_im(x_im) + self.fc_pose(x_pose)
        x = F.leaky_relu(x, negative_slope=self.relu_coeff)
        if drop_rate > 0:
            x = F.dropout(x, p=drop_rate, training=self.training)
        return x


class GQCNNTorch(nn.Module):
    """GQ-CNN network implemented in PyTorch."""

    def __init__(self, gqcnn_config, verbose=True, log_file=None):
        """
        Parameters
        ----------
        gqcnn_config : dict
            Python dictionary of model configuration parameters.
        verbose : bool
            Whether or not to log model output to stdout.
        log_file : str
            If provided, model output will also be logged to this file.
        """
        super().__init__()

        self._verbose = verbose
        self._config = gqcnn_config
        self._parse_config(gqcnn_config)
        self._build_network()

        # Initialize means and stds
        self._im_mean = 0.0
        self._im_std = 1.0
        self._pose_mean = np.zeros(self._pose_dim)
        self._pose_std = np.ones(self._pose_dim)
        self._im_depth_sub_mean = 0.0
        self._im_depth_sub_std = 1.0

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def load(model_dir, verbose=True, log_file=None):
        """Instantiate a trained GQ-CNN for inference.

        Parameters
        ----------
        model_dir : str
            Path to trained GQ-CNN model.
        verbose : bool
            Whether or not to log model output to stdout.
        log_file : str
            If provided, model output will also be logged to this file.

        Returns
        -------
        GQCNNTorch
            Initialized GQ-CNN.
        """
        config_file = os.path.join(model_dir, GQCNNFilenames.SAVED_CFG)
        with open(config_file) as data_file:
            train_config = json.load(data_file, object_pairs_hook=OrderedDict)

        # Support for legacy configs
        try:
            gqcnn_config = train_config["gqcnn"]
        except KeyError:
            gqcnn_config = train_config["gqcnn_config"]
            gqcnn_config["debug"] = 0
            gqcnn_config["seed"] = 0
            gqcnn_config["num_angular_bins"] = 0
            gqcnn_config["input_depth_mode"] = InputDepthMode.POSE_STREAM

            # Convert legacy architecture format
            arch_config = gqcnn_config["architecture"]
            if "im_stream" not in arch_config:
                gqcnn_config["architecture"] = _convert_legacy_architecture(arch_config)

        # Initialize model
        gqcnn = GQCNNTorch(gqcnn_config, verbose=verbose, log_file=log_file)

        # Load weights
        pytorch_model_path = os.path.join(model_dir, GQCNNFilenames.FINAL_MODEL_PYTORCH)
        if os.path.exists(pytorch_model_path):
            gqcnn.load_weights(pytorch_model_path)
        else:
            # Try to convert from TensorFlow
            tf_ckpt_path = os.path.join(model_dir, GQCNNFilenames.FINAL_MODEL)
            if os.path.exists(tf_ckpt_path + ".index") or os.path.exists(tf_ckpt_path):
                if verbose:
                    logger.info("Converting TensorFlow weights to PyTorch...")
                gqcnn._load_tf_weights(tf_ckpt_path)

        # Load mean and std
        gqcnn.init_mean_and_std(model_dir)

        # Setup for inference
        training_mode = train_config.get("training_mode", TrainingMode.CLASSIFICATION)
        if training_mode == TrainingMode.CLASSIFICATION:
            gqcnn._add_softmax = True
        else:
            gqcnn._add_softmax = False

        gqcnn.eval()
        return gqcnn

    def _parse_config(self, gqcnn_config):
        """Parse configuration file."""
        self._batch_size = gqcnn_config["batch_size"]
        self._train_im_height = gqcnn_config["im_height"]
        self._train_im_width = gqcnn_config["im_width"]
        self._im_height = self._train_im_height
        self._im_width = self._train_im_width
        self._num_channels = gqcnn_config["im_channels"]

        try:
            self._gripper_mode = gqcnn_config["gripper_mode"]
        except KeyError:
            # Legacy support
            self._input_data_mode = gqcnn_config.get("input_data_mode", "parallel_jaw")
            mode_map = {
                "tf_image": GripperMode.LEGACY_PARALLEL_JAW,
                "tf_image_suction": GripperMode.LEGACY_SUCTION,
                "suction": GripperMode.SUCTION,
                "multi_suction": GripperMode.MULTI_SUCTION,
                "parallel_jaw": GripperMode.PARALLEL_JAW,
            }
            self._gripper_mode = mode_map.get(
                self._input_data_mode, GripperMode.PARALLEL_JAW
            )

        self._pose_dim = pose_dim(self._gripper_mode)
        self._architecture = gqcnn_config["architecture"]
        self._input_depth_mode = gqcnn_config.get(
            "input_depth_mode", InputDepthMode.POSE_STREAM
        )

        # Normalization parameters
        self._normalization_radius = gqcnn_config.get("radius", 2)
        self._normalization_alpha = gqcnn_config.get("alpha", 2e-5)
        self._normalization_beta = gqcnn_config.get("beta", 0.75)
        self._normalization_bias = gqcnn_config.get("bias", 1.0)

        self._relu_coeff = gqcnn_config.get("relu_coeff", 0.0)
        self._debug = gqcnn_config.get("debug", False)
        self._rand_seed = gqcnn_config.get("seed", 0)

        self._angular_bins = gqcnn_config.get("angular_bins", 0)
        self._max_angle = np.deg2rad(gqcnn_config.get("max_angle", 180))

        self._add_softmax = False

    def _build_network(self):
        """Build the network architecture."""
        self._normalization_params = {
            "radius": self._normalization_radius,
            "alpha": self._normalization_alpha,
            "beta": self._normalization_beta,
            "bias": self._normalization_bias,
        }

        # Build image stream
        self._im_stream = nn.ModuleDict()
        self._pose_stream = nn.ModuleDict()
        self._merge_stream = nn.ModuleDict()

        im_arch = self._architecture.get("im_stream", {})
        pose_arch = self._architecture.get("pose_stream", {})
        merge_arch = self._architecture.get("merge_stream", {})

        # Track dimensions for building layers
        in_channels = self._num_channels
        in_height = self._im_height
        in_width = self._im_width
        fan_in_im = None

        # Build image stream
        for layer_name, layer_config in im_arch.items():
            layer_type = layer_config["type"]

            if layer_type == "conv":
                self._im_stream[layer_name] = ConvBlock(
                    in_channels=in_channels,
                    out_channels=layer_config["num_filt"],
                    kernel_size=layer_config["filt_dim"],
                    pool_size=layer_config["pool_size"],
                    pool_stride=layer_config["pool_stride"],
                    relu_coeff=self._relu_coeff,
                    norm=layer_config.get("norm", False),
                    padding=layer_config.get("pad", "same"),
                    normalization_params=self._normalization_params,
                )

                # Update dimensions
                if layer_config.get("pad", "same").lower() == "same":
                    in_height = (in_height + layer_config["pool_stride"] - 1) // layer_config["pool_stride"]
                    in_width = (in_width + layer_config["pool_stride"] - 1) // layer_config["pool_stride"]
                else:
                    in_height = math.ceil(
                        (in_height - layer_config["filt_dim"] + 1)
                        / layer_config["pool_stride"]
                    )
                    in_width = math.ceil(
                        (in_width - layer_config["filt_dim"] + 1)
                        / layer_config["pool_stride"]
                    )
                in_channels = layer_config["num_filt"]

            elif layer_type == "fc":
                if layer_config.get("out_size", 0) == 0:
                    continue
                if fan_in_im is None:
                    fan_in_im = in_height * in_width * in_channels
                self._im_stream[layer_name] = FCBlock(
                    in_features=fan_in_im,
                    out_features=layer_config["out_size"],
                    relu_coeff=self._relu_coeff,
                    final_layer=False,
                )
                fan_in_im = layer_config["out_size"]

        if fan_in_im is None:
            fan_in_im = in_height * in_width * in_channels

        # Build pose stream
        fan_in_pose = self._pose_dim
        for layer_name, layer_config in pose_arch.items():
            if layer_config["type"] == "pc":
                if layer_config.get("out_size", 0) == 0:
                    continue
                self._pose_stream[layer_name] = PoseBlock(
                    in_features=fan_in_pose,
                    out_features=layer_config["out_size"],
                    relu_coeff=self._relu_coeff,
                )
                fan_in_pose = layer_config["out_size"]

        # Build merge stream
        layers_list = list(merge_arch.items())
        for i, (layer_name, layer_config) in enumerate(layers_list):
            layer_type = layer_config["type"]

            if layer_type == "fc_merge":
                if layer_config.get("out_size", 0) == 0:
                    continue
                self._merge_stream[layer_name] = MergeBlock(
                    in_features_im=fan_in_im,
                    in_features_pose=fan_in_pose,
                    out_features=layer_config["out_size"],
                    relu_coeff=self._relu_coeff,
                )
                fan_in_merge = layer_config["out_size"]

            elif layer_type == "fc":
                if layer_config.get("out_size", 0) == 0:
                    continue
                is_final = i == len(layers_list) - 1
                self._merge_stream[layer_name] = FCBlock(
                    in_features=fan_in_merge,
                    out_features=layer_config["out_size"],
                    relu_coeff=self._relu_coeff,
                    final_layer=is_final,
                )
                fan_in_merge = layer_config["out_size"]

        # Store feature tensors reference
        self._feature_tensors = {}

    def forward(self, images, poses, drop_rate=0.0):
        """Forward pass through the network.

        Parameters
        ----------
        images : torch.Tensor
            Batch of depth images [N, C, H, W].
        poses : torch.Tensor
            Batch of gripper poses [N, pose_dim].
        drop_rate : float
            Dropout rate.

        Returns
        -------
        torch.Tensor
            Network output.
        """
        # Handle depth subtraction mode
        if self._input_depth_mode == InputDepthMode.SUB:
            depth = poses.unsqueeze(-1).unsqueeze(-1)
            images = images - depth
            images = (images - self._im_depth_sub_mean) / self._im_depth_sub_std

        # Image stream
        x_im = images
        for name, layer in self._im_stream.items():
            if isinstance(layer, ConvBlock):
                x_im = layer(x_im)
                self._feature_tensors[name] = x_im
            elif isinstance(layer, FCBlock):
                # Flatten if coming from conv
                if len(x_im.shape) == 4:
                    x_im = x_im.view(x_im.size(0), -1)
                x_im = layer(x_im, drop_rate)
                self._feature_tensors[name] = x_im

        # Flatten if needed
        if len(x_im.shape) == 4:
            x_im = x_im.view(x_im.size(0), -1)

        # Pose stream
        x_pose = poses
        for name, layer in self._pose_stream.items():
            x_pose = layer(x_pose)
            self._feature_tensors[name] = x_pose

        # Merge stream
        x = None
        for name, layer in self._merge_stream.items():
            if isinstance(layer, MergeBlock):
                x = layer(x_im, x_pose, drop_rate)
            else:
                x = layer(x, drop_rate)
            self._feature_tensors[name] = x

        # Add softmax if needed
        if self._add_softmax and x is not None:
            if self._angular_bins > 0:
                # Pairwise softmax
                splits = torch.chunk(x, self._angular_bins, dim=-1)
                x = torch.cat([F.softmax(s, dim=-1) for s in splits], dim=-1)
            else:
                x = F.softmax(x, dim=-1)

        return x

    def init_mean_and_std(self, model_dir):
        """Load means and stds from model directory."""
        if self._input_depth_mode == InputDepthMode.POSE_STREAM:
            try:
                self._im_mean = np.load(
                    os.path.join(model_dir, GQCNNFilenames.IM_MEAN)
                )
                self._im_std = np.load(os.path.join(model_dir, GQCNNFilenames.IM_STD))
            except IOError as e:
                if e.errno == errno.ENOENT:
                    self._im_mean = np.load(
                        os.path.join(model_dir, GQCNNFilenames.LEG_MEAN)
                    )
                    self._im_std = np.load(
                        os.path.join(model_dir, GQCNNFilenames.LEG_STD)
                    )
                else:
                    raise e

            self._pose_mean = np.load(
                os.path.join(model_dir, GQCNNFilenames.POSE_MEAN)
            )
            self._pose_std = np.load(os.path.join(model_dir, GQCNNFilenames.POSE_STD))

            if (
                len(self._pose_mean.shape) > 0
                and self._pose_mean.shape[0] != self._pose_dim
            ):
                if (
                    len(self._pose_mean.shape) > 1
                    and self._pose_mean.shape[1] == self._pose_dim
                ):
                    self._pose_mean = self._pose_mean[0, :]
                    self._pose_std = self._pose_std[0, :]
                else:
                    self._pose_mean = read_pose_data(self._pose_mean, self._gripper_mode)
                    self._pose_std = read_pose_data(self._pose_std, self._gripper_mode)

        elif self._input_depth_mode == InputDepthMode.SUB:
            self._im_depth_sub_mean = np.load(
                os.path.join(model_dir, GQCNNFilenames.IM_DEPTH_SUB_MEAN)
            )
            self._im_depth_sub_std = np.load(
                os.path.join(model_dir, GQCNNFilenames.IM_DEPTH_SUB_STD)
            )
        elif self._input_depth_mode == InputDepthMode.IM_ONLY:
            self._im_mean = np.load(os.path.join(model_dir, GQCNNFilenames.IM_MEAN))
            self._im_std = np.load(os.path.join(model_dir, GQCNNFilenames.IM_STD))

    def load_weights(self, weights_path):
        """Load PyTorch weights from file."""
        state_dict = torch.load(weights_path, map_location=self._device)
        self.load_state_dict(state_dict, strict=False)
        if self._verbose:
            logger.info(f"Loaded weights from {weights_path}")

    def _load_tf_weights(self, ckpt_path):
        """Load weights from TensorFlow checkpoint."""
        from ..utils.tf_converter import convert_tf_to_pytorch

        # This would require TensorFlow to be installed
        logger.warning(
            "TensorFlow weight loading requires TensorFlow. "
            "Please convert weights manually using tf_converter.py"
        )

    def to_device(self, device=None):
        """Move model to device."""
        if device is None:
            device = self._device
        self._device = device
        return self.to(device)

    def predict(self, image_arr, pose_arr, verbose=False):
        """Predict grasp quality.

        Parameters
        ----------
        image_arr : np.ndarray
            4D tensor of depth images [N, H, W, C].
        pose_arr : np.ndarray
            Tensor of gripper poses [N, pose_dim].
        verbose : bool
            Whether to log progress.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        if verbose:
            logger.info("Predicting...")

        start_time = time.time()

        num_images = image_arr.shape[0]
        output_arr = None

        self.eval()
        with torch.no_grad():
            for i in range(0, num_images, self._batch_size):
                end_i = min(i + self._batch_size, num_images)

                # Normalize inputs
                if self._input_depth_mode == InputDepthMode.POSE_STREAM:
                    images = (image_arr[i:end_i] - self._im_mean) / self._im_std
                    poses = (pose_arr[i:end_i] - self._pose_mean) / self._pose_std
                elif self._input_depth_mode == InputDepthMode.SUB:
                    images = image_arr[i:end_i]
                    poses = pose_arr[i:end_i]
                elif self._input_depth_mode == InputDepthMode.IM_ONLY:
                    images = (image_arr[i:end_i] - self._im_mean) / self._im_std
                    poses = pose_arr[i:end_i]

                # Convert to PyTorch tensors
                # Input: [N, H, W, C] -> [N, C, H, W]
                images = np.transpose(images, (0, 3, 1, 2))
                images_t = torch.from_numpy(images.astype(np.float32)).to(self._device)
                poses_t = torch.from_numpy(poses.astype(np.float32)).to(self._device)

                # Forward pass
                output = self(images_t, poses_t)
                output = output.cpu().numpy()

                if output_arr is None:
                    output_arr = np.zeros([num_images] + list(output.shape[1:]))
                output_arr[i:end_i] = output[: end_i - i]

        pred_time = time.time() - start_time
        if verbose:
            logger.info(f"Prediction took {pred_time:.3f} seconds.")

        return output_arr

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def im_height(self):
        return self._im_height

    @property
    def im_width(self):
        return self._im_width

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def pose_dim(self):
        return self._pose_dim

    @property
    def gripper_mode(self):
        return self._gripper_mode

    @property
    def angular_bins(self):
        return self._angular_bins

    @property
    def max_angle(self):
        return self._max_angle

    @property
    def input_depth_mode(self):
        return self._input_depth_mode

    @property
    def stride(self):
        return reduce(
            operator.mul,
            [
                layer_config["pool_stride"]
                for layer_config in self._architecture.get("im_stream", {}).values()
                if layer_config["type"] == "conv"
            ],
            1,
        )


def _convert_legacy_architecture(arch_config):
    """Convert legacy architecture format to new format."""
    new_arch_config = OrderedDict()
    new_arch_config["im_stream"] = OrderedDict()
    new_arch_config["pose_stream"] = OrderedDict()
    new_arch_config["merge_stream"] = OrderedDict()

    conv_layers = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2"]
    for layer_name in conv_layers:
        if layer_name in arch_config:
            new_arch_config["im_stream"][layer_name] = arch_config[layer_name].copy()
            new_arch_config["im_stream"][layer_name]["type"] = "conv"
            new_arch_config["im_stream"][layer_name]["pad"] = arch_config[
                layer_name
            ].get("padding", "SAME")

    if "fc3" in arch_config:
        new_arch_config["im_stream"]["fc3"] = arch_config["fc3"].copy()
        new_arch_config["im_stream"]["fc3"]["type"] = "fc"

    for layer_name in ["pc1", "pc2"]:
        if layer_name in arch_config:
            new_arch_config["pose_stream"][layer_name] = arch_config[layer_name].copy()
            new_arch_config["pose_stream"][layer_name]["type"] = "pc"

    if "fc4" in arch_config:
        new_arch_config["merge_stream"]["fc4"] = arch_config["fc4"].copy()
        new_arch_config["merge_stream"]["fc4"]["type"] = "fc_merge"

    if "fc5" in arch_config:
        new_arch_config["merge_stream"]["fc5"] = arch_config["fc5"].copy()
        new_arch_config["merge_stream"]["fc5"]["type"] = "fc"

    return new_arch_config
