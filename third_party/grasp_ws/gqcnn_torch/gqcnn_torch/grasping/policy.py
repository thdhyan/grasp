# -*- coding: utf-8 -*-
"""
Grasp planning policies using GQ-CNN.
Based on the original gqcnn implementation.
"""
import numpy as np
import logging
from abc import ABC, abstractmethod

from .grasp import GraspAction, Grasp2D
from .image_grasp_sampler import ImageGraspSamplerFactory
from .grasp_quality_function import GQCNNQualityFunction
from ..utils import NoValidGraspsException

logger = logging.getLogger(__name__)


class GraspingPolicy(ABC):
    """Abstract base class for grasping policies."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Policy configuration.
        """
        self._config = config

    @abstractmethod
    def action(self, state):
        """Plan a grasp action.

        Parameters
        ----------
        state : RgbdImageState
            Current state.

        Returns
        -------
        GraspAction
            Planned grasp action.
        """
        pass


class RobustGraspingPolicy(GraspingPolicy):
    """Robust grasping policy using iterative optimization."""

    def __init__(self, config, gqcnn):
        """
        Parameters
        ----------
        config : dict
            Policy configuration.
        gqcnn : GQCNNTorch
            Trained GQ-CNN model.
        """
        super().__init__(config)
        self._gqcnn = gqcnn

        # Parse config
        self._num_seed_samples = config.get("num_seed_samples", 128)
        self._num_gmm_samples = config.get("num_gmm_samples", 64)
        self._num_iters = config.get("num_iters", 3)
        self._gmm_refit_p = config.get("gmm_refit_p", 0.25)
        self._gmm_component_frac = config.get("gmm_component_frac", 0.4)
        self._gmm_reg_covar = config.get("gmm_reg_covar", 0.01)

        # Sampling config
        sampling_config = config.get("sampling", {})
        self._sampler_type = sampling_config.get("type", "antipodal_depth")
        self._sampler = ImageGraspSamplerFactory.sampler(
            self._sampler_type, sampling_config
        )

        # Metric config
        metric_config = config.get("metric", {})
        self._quality_fn = GQCNNQualityFunction(gqcnn, metric_config)

    def action(self, state):
        """Plan a robust grasp action.

        Parameters
        ----------
        state : RgbdImageState
            Current state.

        Returns
        -------
        GraspAction
            Best grasp action.
        """
        # Sample seed grasps
        grasps = self._sampler.sample(
            state.rgbd_im,
            state.camera_intr,
            self._num_seed_samples,
            segmask=state.segmask,
        )

        if len(grasps) == 0:
            raise NoValidGraspsException("No valid grasps found during sampling.")

        # Create initial actions
        actions = [GraspAction(g, 0.0) for g in grasps]

        # Iterative refinement
        for iteration in range(self._num_iters):
            # Evaluate quality
            qualities = self._quality_fn.quality(state, actions)

            # Update action q-values
            for action, q in zip(actions, qualities):
                action._q_value = q

            # Select top grasps for GMM fitting
            sorted_indices = np.argsort(qualities)[::-1]
            num_elite = max(1, int(self._gmm_refit_p * len(actions)))
            elite_indices = sorted_indices[:num_elite]

            # Stop if this is the last iteration
            if iteration == self._num_iters - 1:
                break

            # Fit GMM to elite grasps
            elite_grasps = [grasps[i] for i in elite_indices]
            
            try:
                new_grasps = self._sample_from_gmm(elite_grasps, state)
                grasps = new_grasps
                actions = [GraspAction(g, 0.0) for g in grasps]
            except Exception as e:
                logger.warning(f"GMM sampling failed: {e}. Using current grasps.")
                break

        # Return best grasp
        best_idx = np.argmax([a.q_value for a in actions])
        return actions[best_idx]

    def _sample_from_gmm(self, elite_grasps, state):
        """Sample new grasps from GMM fitted to elite grasps.

        Parameters
        ----------
        elite_grasps : list
            List of elite Grasp2D objects.
        state : RgbdImageState
            Current state.

        Returns
        -------
        list
            New sampled grasps.
        """
        from sklearn.mixture import GaussianMixture

        # Extract features from elite grasps
        features = np.array([g.feature_vec() for g in elite_grasps])

        # Fit GMM
        n_components = max(1, int(self._gmm_component_frac * len(elite_grasps)))
        n_components = min(n_components, len(elite_grasps))

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=self._gmm_reg_covar,
        )
        gmm.fit(features)

        # Sample new grasps
        new_features, _ = gmm.sample(self._num_gmm_samples)

        # Convert back to grasps
        new_grasps = []
        for feat in new_features:
            grasp = Grasp2D(
                center=feat[:2],
                angle=feat[2],
                depth=feat[3],
                camera_intr=state.camera_intr,
            )
            new_grasps.append(grasp)

        return new_grasps


class UniformRandomGraspingPolicy(GraspingPolicy):
    """Policy that samples grasps uniformly at random."""

    def __init__(self, config, gqcnn=None):
        """
        Parameters
        ----------
        config : dict
            Policy configuration.
        gqcnn : GQCNNTorch, optional
            Trained GQ-CNN model (for quality evaluation).
        """
        super().__init__(config)
        self._gqcnn = gqcnn

        # Sampling config
        sampling_config = config.get("sampling", {})
        self._sampler_type = sampling_config.get("type", "antipodal_depth")
        self._sampler = ImageGraspSamplerFactory.sampler(
            self._sampler_type, sampling_config
        )
        self._num_samples = config.get("num_samples", 100)

        if gqcnn is not None:
            metric_config = config.get("metric", {})
            self._quality_fn = GQCNNQualityFunction(gqcnn, metric_config)
        else:
            self._quality_fn = None

    def action(self, state):
        """Sample a random grasp.

        Parameters
        ----------
        state : RgbdImageState
            Current state.

        Returns
        -------
        GraspAction
            Random grasp action.
        """
        grasps = self._sampler.sample(
            state.rgbd_im,
            state.camera_intr,
            self._num_samples,
            segmask=state.segmask,
        )

        if len(grasps) == 0:
            raise NoValidGraspsException("No valid grasps found during sampling.")

        # Select random grasp
        idx = np.random.randint(len(grasps))
        grasp = grasps[idx]

        # Evaluate quality if model available
        if self._quality_fn is not None:
            action = GraspAction(grasp, 0.0)
            qualities = self._quality_fn.quality(state, [action])
            q_value = qualities[0]
        else:
            q_value = 0.0

        return GraspAction(grasp, q_value)


class CrossEntropyRobustGraspingPolicy(RobustGraspingPolicy):
    """Cross-entropy method for robust grasp planning."""

    def __init__(self, config, gqcnn):
        super().__init__(config, gqcnn)


class FullyConvolutionalGraspingPolicyParallelJaw(GraspingPolicy):
    """FC-GQ-CNN based policy for parallel jaw grasping."""

    def __init__(self, config, fc_gqcnn):
        """
        Parameters
        ----------
        config : dict
            Policy configuration.
        fc_gqcnn : FCGQCNNTorch
            Trained FC-GQ-CNN model.
        """
        super().__init__(config)
        self._fc_gqcnn = fc_gqcnn

    def action(self, state):
        """Plan grasp using fully convolutional inference.

        Parameters
        ----------
        state : RgbdImageState
            Current state.

        Returns
        -------
        GraspAction
            Best grasp action.
        """
        depth_im = state.depth_im
        
        # Dummy pose for FC-GQ-CNN
        pose = np.zeros((1, self._fc_gqcnn.pose_dim))

        # Get spatial quality predictions
        predictions = self._fc_gqcnn.predict(
            depth_im[np.newaxis, ...],
            pose,
        )

        # predictions shape: [1, H', W', 2] for binary classification
        pred_map = predictions[0]

        if pred_map.shape[-1] == 2:
            quality_map = pred_map[:, :, 1]
        else:
            quality_map = pred_map[:, :, 0]

        # Apply segmask if available
        if state.segmask is not None:
            # Resize segmask to match quality map
            import cv2
            resized_mask = cv2.resize(
                state.segmask.astype(np.float32),
                (quality_map.shape[1], quality_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            quality_map = quality_map * resized_mask

        # Find best grasp location
        best_idx = np.unravel_index(np.argmax(quality_map), quality_map.shape)
        best_quality = quality_map[best_idx]

        # Convert to image coordinates
        stride = self._fc_gqcnn.stride
        center_y = best_idx[0] * stride + stride // 2
        center_x = best_idx[1] * stride + stride // 2
        center = np.array([center_x, center_y])

        # Estimate depth at grasp point
        depth = depth_im[
            min(center_y, depth_im.shape[0] - 1),
            min(center_x, depth_im.shape[1] - 1),
            0,
        ]

        grasp = Grasp2D(
            center=center,
            angle=0.0,  # FC-GQ-CNN doesn't predict angle directly
            depth=depth,
            camera_intr=state.camera_intr,
        )

        return GraspAction(grasp, best_quality)


class FullyConvolutionalGraspingPolicySuction(FullyConvolutionalGraspingPolicyParallelJaw):
    """FC-GQ-CNN based policy for suction grasping."""

    def action(self, state):
        """Plan suction grasp using fully convolutional inference."""
        from .grasp import SuctionPoint2D

        action = super().action(state)

        # Convert to suction grasp
        suction_grasp = SuctionPoint2D(
            center=action.grasp.center,
            axis=np.array([0, 0, 1]),  # Vertical approach
            depth=action.grasp.depth,
            camera_intr=action.grasp.camera_intr,
        )

        return GraspAction(suction_grasp, action.q_value)
