# -*- coding: utf-8 -*-
"""
Grasp representation classes.
Based on the original gqcnn implementation.
"""
import numpy as np


class Grasp2D:
    """2D grasp representation in image space."""

    def __init__(self, center, angle, depth, width=0.0, camera_intr=None):
        """
        Parameters
        ----------
        center : np.ndarray
            2D center point [x, y] in image coordinates.
        angle : float
            Grasp angle in radians.
        depth : float
            Grasp depth.
        width : float
            Grasp width.
        camera_intr : object, optional
            Camera intrinsics.
        """
        self._center = np.array(center)
        self._angle = angle
        self._depth = depth
        self._width = width
        self._camera_intr = camera_intr

    @property
    def center(self):
        return self._center

    @property
    def angle(self):
        return self._angle

    @property
    def depth(self):
        return self._depth

    @property
    def width(self):
        return self._width

    @property
    def camera_intr(self):
        return self._camera_intr

    @property
    def axis(self):
        """Grasp axis direction in image space."""
        return np.array([np.cos(self._angle), np.sin(self._angle)])

    def feature_vec(self):
        """Return feature vector [x, y, angle, depth]."""
        return np.array([self._center[0], self._center[1], self._angle, self._depth])

    def __repr__(self):
        return (
            f"Grasp2D(center={self._center}, angle={self._angle:.3f}, "
            f"depth={self._depth:.3f})"
        )


class SuctionPoint2D:
    """2D suction grasp representation."""

    def __init__(self, center, axis, depth, camera_intr=None):
        """
        Parameters
        ----------
        center : np.ndarray
            2D center point [x, y] in image coordinates.
        axis : np.ndarray
            Approach axis direction.
        depth : float
            Grasp depth.
        camera_intr : object, optional
            Camera intrinsics.
        """
        self._center = np.array(center)
        self._axis = np.array(axis)
        self._depth = depth
        self._camera_intr = camera_intr

    @property
    def center(self):
        return self._center

    @property
    def axis(self):
        return self._axis

    @property
    def depth(self):
        return self._depth

    @property
    def angle(self):
        """Compute approach angle from axis."""
        return np.arctan2(self._axis[1], self._axis[0])

    @property
    def camera_intr(self):
        return self._camera_intr

    def feature_vec(self):
        """Return feature vector."""
        return np.array([
            self._center[0], self._center[1],
            self._depth, self._axis[0], self._axis[1]
        ])

    def __repr__(self):
        return (
            f"SuctionPoint2D(center={self._center}, depth={self._depth:.3f})"
        )


class GraspAction:
    """Grasp action with quality score."""

    def __init__(self, grasp, q_value, image=None):
        """
        Parameters
        ----------
        grasp : Grasp2D or SuctionPoint2D
            Grasp representation.
        q_value : float
            Predicted grasp quality.
        image : np.ndarray, optional
            Associated image.
        """
        self._grasp = grasp
        self._q_value = q_value
        self._image = image

    @property
    def grasp(self):
        return self._grasp

    @property
    def q_value(self):
        return self._q_value

    @property
    def image(self):
        return self._image

    def __repr__(self):
        return f"GraspAction(grasp={self._grasp}, q_value={self._q_value:.3f})"


class RgbdImageState:
    """State representation with RGB-D image."""

    def __init__(
        self,
        rgbd_im,
        camera_intr,
        segmask=None,
        obj_segmask=None,
        full_observed=True,
    ):
        """
        Parameters
        ----------
        rgbd_im : np.ndarray
            RGB-D image (H, W, 4).
        camera_intr : object
            Camera intrinsics.
        segmask : np.ndarray, optional
            Segmentation mask.
        obj_segmask : np.ndarray, optional
            Object segmentation mask.
        full_observed : bool
            Whether the scene is fully observed.
        """
        self._rgbd_im = rgbd_im
        self._camera_intr = camera_intr
        self._segmask = segmask
        self._obj_segmask = obj_segmask
        self._full_observed = full_observed

    @property
    def rgbd_im(self):
        return self._rgbd_im

    @property
    def camera_intr(self):
        return self._camera_intr

    @property
    def segmask(self):
        return self._segmask

    @property
    def obj_segmask(self):
        return self._obj_segmask

    @property
    def full_observed(self):
        return self._full_observed

    @property
    def depth_im(self):
        """Get depth image."""
        return self._rgbd_im[:, :, 3:4]

    @property
    def color_im(self):
        """Get color image."""
        return self._rgbd_im[:, :, :3]
