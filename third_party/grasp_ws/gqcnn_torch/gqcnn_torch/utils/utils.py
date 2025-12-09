# -*- coding: utf-8 -*-
"""
Utility functions for gqcnn_torch package.
Ported from TensorFlow implementation.
"""
from functools import reduce
import os
import logging

import numpy as np
import skimage.transform as skt

from .enums import GripperMode

# Set up logger
logger = logging.getLogger(__name__)


def set_cuda_visible_devices(gpu_list):
    """Sets CUDA_VISIBLE_DEVICES environment variable.

    Parameters
    ----------
    gpu_list : list
        List of gpus to set as visible.
    """
    if len(gpu_list) == 0:
        return

    cuda_visible_devices = ",".join(str(gpu) for gpu in gpu_list)
    logger.info(f"Setting CUDA_VISIBLE_DEVICES = {cuda_visible_devices}")
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def pose_dim(gripper_mode):
    """Returns the dimensions of the pose vector for the given gripper mode.

    Parameters
    ----------
    gripper_mode : str
        Gripper mode enum value.

    Returns
    -------
    int
        Pose dimension for the gripper mode.
    """
    if gripper_mode == GripperMode.PARALLEL_JAW:
        return 1
    elif gripper_mode == GripperMode.SUCTION:
        return 2
    elif gripper_mode == GripperMode.MULTI_SUCTION:
        return 1
    elif gripper_mode == GripperMode.LEGACY_PARALLEL_JAW:
        return 1
    elif gripper_mode == GripperMode.LEGACY_SUCTION:
        return 2
    else:
        raise ValueError(f"Gripper mode '{gripper_mode}' not supported.")


def read_pose_data(pose_arr, gripper_mode):
    """Read pose data and slice according to gripper mode.

    Parameters
    ----------
    pose_arr : np.ndarray
        Full pose data array.
    gripper_mode : str
        Gripper mode enum value.

    Returns
    -------
    np.ndarray
        Sliced pose data.
    """
    if gripper_mode == GripperMode.PARALLEL_JAW:
        if pose_arr.ndim == 1:
            return pose_arr[2:3]
        else:
            return pose_arr[:, 2:3]
    elif gripper_mode == GripperMode.SUCTION:
        if pose_arr.ndim == 1:
            return np.r_[pose_arr[2], pose_arr[4]]
        else:
            return np.c_[pose_arr[:, 2], pose_arr[:, 4]]
    elif gripper_mode == GripperMode.MULTI_SUCTION:
        if pose_arr.ndim == 1:
            return pose_arr[2:3]
        else:
            return pose_arr[:, 2:3]
    elif gripper_mode == GripperMode.LEGACY_PARALLEL_JAW:
        if pose_arr.ndim == 1:
            return pose_arr[2:3]
        else:
            return pose_arr[:, 2:3]
    elif gripper_mode == GripperMode.LEGACY_SUCTION:
        if pose_arr.ndim == 1:
            return pose_arr[2:4]
        else:
            return pose_arr[:, 2:4]
    else:
        raise ValueError(f"Gripper mode '{gripper_mode}' not supported.")


def reduce_shape(shape):
    """Get shape of a layer for flattening.

    Parameters
    ----------
    shape : tuple
        Layer shape.

    Returns
    -------
    int
        Flattened size.
    """
    if isinstance(shape, (list, tuple)):
        shape_list = list(shape[1:])
    else:
        shape_list = list(shape)[1:]
    
    f = lambda x, y: 1 if y is None else x * y
    return reduce(f, shape_list, 1)


def weight_name_to_layer_name(weight_name):
    """Convert weight name to layer name.

    Parameters
    ----------
    weight_name : str
        Weight variable name.

    Returns
    -------
    str
        Layer name.
    """
    tokens = weight_name.split("_")
    type_name = tokens[-1]

    if type_name in ("weights", "bias"):
        if len(tokens) >= 3 and tokens[-3] == "input":
            return weight_name[:weight_name.rfind("input") - 1]
        return weight_name[:weight_name.rfind(type_name) - 1]
    if type_name == "im":
        return weight_name[:-4]
    if type_name == "pose":
        return weight_name[:-6]
    return weight_name[:-1]


def imresize(image, size, interp="nearest"):
    """Resize image using skimage.

    Parameters
    ----------
    image : np.ndarray
        Image to resize.
    size : int, float, or tuple
        Target size.
    interp : str
        Interpolation method.

    Returns
    -------
    np.ndarray
        Resized image.
    """
    skt_interp_map = {
        "nearest": 0,
        "bilinear": 1,
        "biquadratic": 2,
        "bicubic": 3,
        "biquartic": 4,
        "biquintic": 5
    }
    if interp in ("lanczos", "cubic"):
        raise ValueError("'lanczos' and 'cubic' interpolation not supported.")
    assert interp in skt_interp_map, f"Interpolation '{interp}' not supported."

    if isinstance(size, (tuple, list)):
        output_shape = size
    elif isinstance(size, float):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size
        output_shape = tuple(np_shape.astype(int))
    elif isinstance(size, int):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size / 100.0
        output_shape = tuple(np_shape.astype(int))
    else:
        raise ValueError(f"Invalid type for size '{type(size)}'.")

    return skt.resize(
        image,
        output_shape,
        order=skt_interp_map[interp],
        anti_aliasing=False,
        mode="constant"
    )
