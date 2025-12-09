# -*- coding: utf-8 -*-
"""
Factory functions to obtain GQCNN/FCGQCNN classes.
"""
import logging

from .network_torch import GQCNNTorch
from .fc_network_torch import FCGQCNNTorch

logger = logging.getLogger(__name__)


def get_gqcnn_model(backend="torch", verbose=True):
    """Get the GQ-CNN model for the provided backend.

    Parameters
    ----------
    backend : str
        The backend to use. Supported: "torch", "pytorch".
    verbose : bool
        Whether or not to log initialization output.

    Returns
    -------
    type
        GQ-CNN model class.
    """
    if backend in ("torch", "pytorch"):
        if verbose:
            logger.info("Initializing GQ-CNN with PyTorch as backend...")
        return GQCNNTorch
    else:
        raise ValueError(f"Invalid backend: {backend}. Use 'torch' or 'pytorch'.")


def get_fc_gqcnn_model(backend="torch", verbose=True):
    """Get the FC-GQ-CNN model for the provided backend.

    Parameters
    ----------
    backend : str
        The backend to use. Supported: "torch", "pytorch".
    verbose : bool
        Whether or not to log initialization output.

    Returns
    -------
    type
        FC-GQ-CNN model class.
    """
    if backend in ("torch", "pytorch"):
        if verbose:
            logger.info("Initializing FC-GQ-CNN with PyTorch as backend...")
        return FCGQCNNTorch
    else:
        raise ValueError(f"Invalid backend: {backend}. Use 'torch' or 'pytorch'.")
