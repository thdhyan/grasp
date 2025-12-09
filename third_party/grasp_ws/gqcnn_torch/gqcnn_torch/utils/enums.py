# -*- coding: utf-8 -*-
"""
Constants and enums for gqcnn_torch package.
Ported from TensorFlow implementation.
"""
import math


class GeneralConstants:
    """General constants used throughout the package."""
    SEED = 3472134
    SEED_SAMPLE_MAX = 2**32 - 1
    MAX_PREFETCH_Q_SIZE = 250
    NUM_PREFETCH_Q_WORKERS = 3
    QUEUE_SLEEP = 0.001
    PI = math.pi
    FIGSIZE = 16


class ImageMode:
    """Enum for image modalities."""
    BINARY = "binary"
    DEPTH = "depth"
    BINARY_TF = "binary_tf"
    COLOR_TF = "color_tf"
    GRAY_TF = "gray_tf"
    DEPTH_TF = "depth_tf"
    DEPTH_TF_TABLE = "depth_tf_table"
    TF_DEPTH_IMS = "tf_depth_ims"


class TrainingMode:
    """Enum for training modes."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class GripperMode:
    """Enum for input pose data formats."""
    PARALLEL_JAW = "parallel_jaw"
    SUCTION = "suction"
    MULTI_SUCTION = "multi_suction"
    LEGACY_PARALLEL_JAW = "legacy_parallel_jaw"
    LEGACY_SUCTION = "legacy_suction"


class InputDepthMode:
    """Enum for input depth mode."""
    POSE_STREAM = "pose_stream"
    SUB = "im_depth_sub"
    IM_ONLY = "im_only"


class GQCNNTrainingStatus:
    """Enum for training status."""
    NOT_STARTED = "not_started"
    SETTING_UP = "setting_up"
    TRAINING = "training"


class GQCNNFilenames:
    """Enum for standard filenames."""
    PCT_POS_VAL = "pct_pos_val.npy"
    PCT_POS_TRAIN = "pct_pos_train.npy"
    LEARNING_RATES = "learning_rates.npy"

    TRAIN_ITERS = "train_eval_iters.npy"
    TRAIN_LOSSES = "train_losses.npy"
    TRAIN_ERRORS = "train_errors.npy"
    TOTAL_TRAIN_LOSSES = "total_train_losses.npy"
    TOTAL_TRAIN_ERRORS = "total_train_errors.npy"

    VAL_ITERS = "val_eval_iters.npy"
    VAL_LOSSES = "val_losses.npy"
    VAL_ERRORS = "val_errors.npy"

    LEG_MEAN = "mean.npy"
    LEG_STD = "std.npy"
    IM_MEAN = "im_mean.npy"
    IM_STD = "im_std.npy"
    IM_DEPTH_SUB_MEAN = "im_depth_sub_mean.npy"
    IM_DEPTH_SUB_STD = "im_depth_sub_std.npy"
    POSE_MEAN = "pose_mean.npy"
    POSE_STD = "pose_std.npy"

    FINAL_MODEL = "model.ckpt"
    FINAL_MODEL_PYTORCH = "model.pt"
    INTER_MODEL = "model_{}.ckpt"

    SAVED_ARCH = "architecture.json"
    SAVED_CFG = "config.json"
