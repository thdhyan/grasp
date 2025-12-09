# -*- coding: utf-8 -*-
"""
Factory functions to obtain GQCNNTrainer class.
"""
import logging

from .trainer_torch import GQCNNTrainerTorch

logger = logging.getLogger(__name__)


def get_gqcnn_trainer(backend="torch"):
    """Get the GQ-CNN Trainer for the provided backend.

    Parameters
    ----------
    backend : str
        The backend to use. Supported: "torch", "pytorch".

    Returns
    -------
    type
        GQ-CNN Trainer class.
    """
    if backend in ("torch", "pytorch"):
        return GQCNNTrainerTorch
    else:
        raise ValueError(f"Invalid backend: {backend}. Use 'torch' or 'pytorch'.")
