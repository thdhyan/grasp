# -*- coding: utf-8 -*-
"""
Grasp sampling utilities.
Based on the original gqcnn implementation.
"""
import numpy as np
from abc import ABC, abstractmethod


class GraspSampler(ABC):
    """Abstract base class for grasp samplers."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Sampler configuration.
        """
        self._config = config

    @abstractmethod
    def sample(self, rgbd_im, camera_intr, num_samples, segmask=None):
        """Sample grasps from an RGBD image.

        Parameters
        ----------
        rgbd_im : np.ndarray
            RGB-D image.
        camera_intr : object
            Camera intrinsics.
        num_samples : int
            Number of grasps to sample.
        segmask : np.ndarray, optional
            Segmentation mask.

        Returns
        -------
        list
            List of sampled grasps.
        """
        pass


class AntipodalDepthImageGraspSampler(GraspSampler):
    """Sample antipodal grasps from depth images."""

    def __init__(self, config):
        super().__init__(config)

        # Sampling params
        self._friction_coef = config.get("friction_coef", 1.0)
        self._depth_grad_thresh = config.get("depth_grad_thresh", 0.0025)
        self._depth_grad_gaussian_sigma = config.get("depth_grad_gaussian_sigma", 1.0)
        self._downsample_rate = config.get("downsample_rate", 4)
        self._max_rejection_samples = config.get("max_rejection_samples", 4000)

        # Distance params
        self._max_dist_from_center = config.get("max_dist_from_center", 160)
        self._min_dist_from_boundary = config.get("min_dist_from_boundary", 45)
        self._min_grasp_dist = config.get("min_grasp_dist", 2.5)
        self._angle_dist_weight = config.get("angle_dist_weight", 5.0)

        # Depth sampling
        self._depth_sampling_mode = config.get("depth_sampling_mode", "uniform")
        self._depth_samples_per_grasp = config.get("depth_samples_per_grasp", 3)
        self._min_depth_offset = config.get("min_depth_offset", 0.015)
        self._max_depth_offset = config.get("max_depth_offset", 0.05)

    def sample(self, rgbd_im, camera_intr, num_samples, segmask=None):
        """Sample antipodal grasps from depth image.

        Parameters
        ----------
        rgbd_im : np.ndarray
            RGB-D image (H, W, 4).
        camera_intr : object
            Camera intrinsics.
        num_samples : int
            Number of grasps to sample.
        segmask : np.ndarray, optional
            Segmentation mask.

        Returns
        -------
        list
            List of Grasp2D objects.
        """
        from .grasp import Grasp2D

        depth_im = rgbd_im[:, :, 3] if rgbd_im.ndim == 3 else rgbd_im

        # Get valid depth points
        if segmask is not None:
            valid_mask = (segmask > 0) & (depth_im > 0)
        else:
            valid_mask = depth_im > 0

        valid_indices = np.where(valid_mask)
        if len(valid_indices[0]) == 0:
            return []

        # Sample grasp centers
        grasps = []
        num_sampled = 0
        num_attempts = 0

        while num_sampled < num_samples and num_attempts < self._max_rejection_samples:
            num_attempts += 1

            # Random center
            idx = np.random.randint(len(valid_indices[0]))
            center_y = valid_indices[0][idx]
            center_x = valid_indices[1][idx]
            center = np.array([center_x, center_y])

            # Check distance from boundary
            if (
                center_x < self._min_dist_from_boundary
                or center_x > depth_im.shape[1] - self._min_dist_from_boundary
                or center_y < self._min_dist_from_boundary
                or center_y > depth_im.shape[0] - self._min_dist_from_boundary
            ):
                continue

            # Random angle
            angle = np.random.uniform(-np.pi / 2, np.pi / 2)

            # Depth at center
            depth = depth_im[center_y, center_x]

            # Add depth offset
            depth_offset = np.random.uniform(
                self._min_depth_offset, self._max_depth_offset
            )
            grasp_depth = depth - depth_offset

            grasp = Grasp2D(
                center=center,
                angle=angle,
                depth=grasp_depth,
                camera_intr=camera_intr,
            )

            # Check minimum distance from existing grasps
            too_close = False
            for existing_grasp in grasps:
                dist = np.linalg.norm(grasp.center - existing_grasp.center)
                if dist < self._min_grasp_dist:
                    too_close = True
                    break

            if not too_close:
                grasps.append(grasp)
                num_sampled += 1

        return grasps


class ImageGraspSamplerFactory:
    """Factory for creating grasp samplers."""

    @staticmethod
    def sampler(sampler_type, config):
        """Create a grasp sampler.

        Parameters
        ----------
        sampler_type : str
            Type of sampler.
        config : dict
            Sampler configuration.

        Returns
        -------
        GraspSampler
            Grasp sampler instance.
        """
        if sampler_type == "antipodal_depth":
            return AntipodalDepthImageGraspSampler(config)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
