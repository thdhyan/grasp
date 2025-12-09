# -*- coding: utf-8 -*-
"""
Grasp quality functions using GQ-CNN.
Based on the original gqcnn implementation.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GQCNNQualityFunction:
    """Grasp quality function using GQ-CNN predictions."""

    def __init__(self, gqcnn, config=None):
        """
        Parameters
        ----------
        gqcnn : GQCNNTorch
            Trained GQ-CNN model.
        config : dict, optional
            Configuration parameters.
        """
        self._gqcnn = gqcnn
        self._config = config or {}

        self._crop_height = self._config.get("crop_height", 96)
        self._crop_width = self._config.get("crop_width", 96)

    def quality(self, state, actions, params=None):
        """Compute grasp quality for a set of actions.

        Parameters
        ----------
        state : RgbdImageState
            Current state with RGBD image.
        actions : list
            List of GraspAction objects.
        params : dict, optional
            Additional parameters.

        Returns
        -------
        np.ndarray
            Quality values for each action.
        """
        if len(actions) == 0:
            return np.array([])

        # Extract depth images centered at grasp points
        depth_im = state.depth_im
        images = []
        poses = []

        for action in actions:
            grasp = action.grasp

            # Extract crop centered at grasp
            crop = self._extract_crop(
                depth_im,
                grasp.center,
                grasp.angle,
                self._crop_height,
                self._crop_width,
            )
            images.append(crop)

            # Pose is grasp depth
            pose = np.array([grasp.depth])
            poses.append(pose)

        images = np.array(images)
        poses = np.array(poses)

        # Predict quality
        predictions = self._gqcnn.predict(images, poses)

        # Extract success probability (second column for binary classification)
        if predictions.shape[1] == 2:
            qualities = predictions[:, 1]
        else:
            qualities = predictions.flatten()

        return qualities

    def _extract_crop(self, depth_im, center, angle, height, width):
        """Extract rotated crop from depth image.

        Parameters
        ----------
        depth_im : np.ndarray
            Depth image (H, W, 1) or (H, W).
        center : np.ndarray
            Crop center [x, y].
        angle : float
            Rotation angle in radians.
        height : int
            Crop height.
        width : int
            Crop width.

        Returns
        -------
        np.ndarray
            Cropped and rotated depth image (height, width, 1).
        """
        import cv2

        if depth_im.ndim == 3:
            depth_im = depth_im[:, :, 0]

        # Rotation matrix
        M = cv2.getRotationMatrix2D(
            (center[0], center[1]),
            np.degrees(angle),
            1.0,
        )

        # Rotate image
        rotated = cv2.warpAffine(
            depth_im,
            M,
            (depth_im.shape[1], depth_im.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Extract crop
        x1 = int(center[0] - width // 2)
        y1 = int(center[1] - height // 2)
        x2 = x1 + width
        y2 = y1 + height

        # Pad if necessary
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - depth_im.shape[1])
        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - depth_im.shape[0])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_im.shape[1], x2)
        y2 = min(depth_im.shape[0], y2)

        crop = rotated[y1:y2, x1:x2]

        if pad_left or pad_right or pad_top or pad_bottom:
            crop = np.pad(
                crop,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="edge",
            )

        return crop[:, :, np.newaxis]
