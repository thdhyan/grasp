# -*- coding: utf-8 -*-
"""
FC-GQ-CNN network implemented in PyTorch.
Based on the TensorFlow implementation from Berkeley Autolab.
"""
from collections import OrderedDict
import json
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_torch import GQCNNTorch
from ..utils import TrainingMode, InputDepthMode, GQCNNFilenames

logger = logging.getLogger(__name__)


class FCGQCNNTorch(GQCNNTorch):
    """FC-GQ-CNN network implemented in PyTorch.

    Note
    ----
    FC-GQ-CNNs are never directly trained, but instead a pre-trained GQ-CNN
    is converted to an FC-GQ-CNN at inference time.
    """

    def __init__(self, gqcnn_config, fc_config, verbose=True, log_file=None):
        """
        Parameters
        ----------
        gqcnn_config : dict
            Python dictionary of pre-trained GQ-CNN model configuration parameters.
        fc_config : dict
            Python dictionary of FC-GQ-CNN model configuration parameters.
        verbose : bool
            Whether or not to log model output to stdout.
        log_file : str
            If provided, model output will also be logged to this file.
        """
        # Parse FC config first to override dimensions
        self._fc_im_width = fc_config["im_width"]
        self._fc_im_height = fc_config["im_height"]

        super().__init__(gqcnn_config, verbose=verbose, log_file=log_file)
        
        # Override image dimensions
        self._im_width = self._fc_im_width
        self._im_height = self._fc_im_height

        # Check that conv layers were trained with VALID padding
        for layer_name, layer_config in self._architecture.get("im_stream", {}).items():
            if layer_config["type"] == "conv":
                if layer_config.get("pad", "SAME").upper() != "VALID":
                    logger.warning(
                        f"GQ-CNN used for FC-GQ-CNN should have VALID padding "
                        f"for conv layers. Found layer: {layer_name} with "
                        f"padding: {layer_config.get('pad', 'SAME')}"
                    )

        # Build fully convolutional versions of FC layers
        self._build_fc_conv_layers()

    @staticmethod
    def load(model_dir, fc_config, log_file=None):
        """Load an FC-GQ-CNN from a pre-trained GQ-CNN.

        Parameters
        ----------
        model_dir : str
            Path to pre-trained GQ-CNN model.
        fc_config : dict
            Python dictionary of FC-GQ-CNN model configuration parameters.
        log_file : str
            If provided, model output will also be logged to this file.

        Returns
        -------
        FCGQCNNTorch
            Initialized FC-GQ-CNN.
        """
        config_file = os.path.join(model_dir, GQCNNFilenames.SAVED_CFG)
        with open(config_file) as data_file:
            train_config = json.load(data_file, object_pairs_hook=OrderedDict)

        gqcnn_config = train_config["gqcnn"]

        # Initialize FC-GQ-CNN
        fcgqcnn = FCGQCNNTorch(gqcnn_config, fc_config, log_file=log_file)

        # Load weights
        pytorch_model_path = os.path.join(model_dir, GQCNNFilenames.FINAL_MODEL_PYTORCH)
        if os.path.exists(pytorch_model_path):
            fcgqcnn.load_weights(pytorch_model_path)
        
        fcgqcnn.init_mean_and_std(model_dir)

        training_mode = train_config.get("training_mode", TrainingMode.CLASSIFICATION)
        if training_mode == TrainingMode.CLASSIFICATION:
            fcgqcnn._add_softmax = True
        else:
            fcgqcnn._add_softmax = False

        fcgqcnn.eval()
        return fcgqcnn

    def _build_fc_conv_layers(self):
        """Convert FC layers to fully convolutional layers."""
        # This creates 1x1 conv equivalents of the FC layers for spatial outputs
        self._fc_conv_layers = nn.ModuleDict()

        # Track the spatial dimensions after conv layers
        im_arch = self._architecture.get("im_stream", {})
        
        in_channels = self._num_channels
        in_height = self._train_im_height
        in_width = self._train_im_width

        for layer_name, layer_config in im_arch.items():
            if layer_config["type"] == "conv":
                if layer_config.get("pad", "same").upper() == "VALID":
                    in_height = (in_height - layer_config["filt_dim"] + 1)
                    in_width = (in_width - layer_config["filt_dim"] + 1)
                in_height = (in_height + layer_config["pool_stride"] - 1) // layer_config["pool_stride"]
                in_width = (in_width + layer_config["pool_stride"] - 1) // layer_config["pool_stride"]
                in_channels = layer_config["num_filt"]

        # Store final conv dimensions (this is the filter size for FC->Conv conversion)
        self._fc_filter_dim = min(in_height, in_width)

    def forward(self, images, poses, drop_rate=0.0):
        """Forward pass for FC-GQ-CNN.

        Parameters
        ----------
        images : torch.Tensor
            Batch of depth images [N, C, H, W] or single large image [1, C, H, W].
        poses : torch.Tensor
            Batch of gripper poses [N, pose_dim] or single pose.
        drop_rate : float
            Dropout rate.

        Returns
        -------
        torch.Tensor
            Network output with spatial dimensions [N, num_classes, H', W'].
        """
        # Handle depth subtraction mode
        if self._input_depth_mode == InputDepthMode.SUB:
            depth = poses.unsqueeze(-1).unsqueeze(-1)
            images = images - depth
            images = (images - self._im_depth_sub_mean) / self._im_depth_sub_std

        # Image stream (conv layers only for spatial output)
        x_im = images
        for name, layer in self._im_stream.items():
            if hasattr(layer, 'conv'):  # ConvBlock
                x_im = layer(x_im)
                self._feature_tensors[name] = x_im

        # Get spatial dimensions
        _, _, h, w = x_im.shape

        # Pose stream
        x_pose = poses
        for name, layer in self._pose_stream.items():
            x_pose = layer(x_pose)
            self._feature_tensors[name] = x_pose

        # Expand pose to spatial dimensions
        # [N, pose_dim] -> [N, pose_dim, H, W]
        x_pose_expanded = x_pose.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)

        # For merge stream, we need to handle it differently for fully convolutional
        # Here we concatenate and use 1x1 convs (simplified approach)
        x = None
        for name, layer in self._merge_stream.items():
            if hasattr(layer, 'fc_im'):  # MergeBlock
                # Flatten spatial dims, process, reshape back
                batch_size = x_im.shape[0]
                x_im_flat = x_im.permute(0, 2, 3, 1).reshape(-1, x_im.shape[1])
                x_pose_flat = x_pose.unsqueeze(1).unsqueeze(1).expand(-1, h, w, -1).reshape(-1, x_pose.shape[1])
                x = layer(x_im_flat, x_pose_flat, drop_rate)
                x = x.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
            else:  # FCBlock
                batch_size, channels, h, w = x.shape
                x_flat = x.permute(0, 2, 3, 1).reshape(-1, channels)
                x = layer(x_flat, drop_rate)
                x = x.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
            self._feature_tensors[name] = x

        # Add softmax if needed
        if self._add_softmax and x is not None:
            if self._angular_bins > 0:
                splits = torch.chunk(x, self._angular_bins, dim=1)
                x = torch.cat([F.softmax(s, dim=1) for s in splits], dim=1)
            else:
                x = F.softmax(x, dim=1)

        return x

    def predict(self, image_arr, pose_arr, verbose=False):
        """Predict grasp quality with spatial outputs.

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
            Predictions with spatial dimensions.
        """
        if verbose:
            logger.info("Predicting (FC-GQ-CNN)...")

        self.eval()
        with torch.no_grad():
            # Normalize inputs
            if self._input_depth_mode == InputDepthMode.POSE_STREAM:
                images = (image_arr - self._im_mean) / self._im_std
                poses = (pose_arr - self._pose_mean) / self._pose_std
            elif self._input_depth_mode == InputDepthMode.SUB:
                images = image_arr
                poses = pose_arr
            elif self._input_depth_mode == InputDepthMode.IM_ONLY:
                images = (image_arr - self._im_mean) / self._im_std
                poses = pose_arr

            # Convert to PyTorch tensors
            # Input: [N, H, W, C] -> [N, C, H, W]
            images = np.transpose(images, (0, 3, 1, 2))
            images_t = torch.from_numpy(images.astype(np.float32)).to(self._device)
            poses_t = torch.from_numpy(poses.astype(np.float32)).to(self._device)

            # Forward pass
            output = self(images_t, poses_t)
            # [N, C, H, W] -> [N, H, W, C]
            output = output.permute(0, 2, 3, 1).cpu().numpy()

        return output
