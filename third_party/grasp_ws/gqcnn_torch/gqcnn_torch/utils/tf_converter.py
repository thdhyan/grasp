# -*- coding: utf-8 -*-
"""
Utility functions for converting TensorFlow weights to PyTorch.
"""
import os
import json
from collections import OrderedDict
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def convert_tf_to_pytorch(tf_model_dir, pytorch_output_path, verbose=True):
    """Convert TensorFlow GQ-CNN checkpoint to PyTorch state dict.

    Parameters
    ----------
    tf_model_dir : str
        Directory containing TensorFlow model checkpoint.
    pytorch_output_path : str
        Path to save PyTorch model.
    verbose : bool
        Whether to log conversion progress.

    Returns
    -------
    dict
        PyTorch state dictionary.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required for weight conversion. "
            "Install with: pip install tensorflow"
        )

    if verbose:
        logger.info(f"Loading TensorFlow checkpoint from {tf_model_dir}")

    # Load checkpoint
    ckpt_file = os.path.join(tf_model_dir, "model.ckpt")
    reader = tf.train.NewCheckpointReader(ckpt_file)

    # Get variable names and shapes
    var_to_shape_map = reader.get_variable_to_shape_map()

    # Convert weights
    pytorch_state_dict = OrderedDict()

    for var_name, shape in var_to_shape_map.items():
        tensor = reader.get_tensor(var_name)
        short_name = var_name.split("/")[-1]

        # Convert TF naming to PyTorch naming
        pytorch_name = _convert_tf_name_to_pytorch(short_name)

        # Handle weight transposition for convolutions
        if "conv" in short_name.lower() and "weight" in short_name.lower():
            # TF: [H, W, C_in, C_out] -> PyTorch: [C_out, C_in, H, W]
            tensor = np.transpose(tensor, (3, 2, 0, 1))
        elif "fc" in short_name.lower() or "pc" in short_name.lower():
            if "weight" in short_name.lower() and len(tensor.shape) == 2:
                # TF: [in, out] -> PyTorch: [out, in]
                tensor = np.transpose(tensor, (1, 0))

        pytorch_state_dict[pytorch_name] = torch.from_numpy(tensor.copy())

        if verbose:
            logger.info(f"Converted {var_name} -> {pytorch_name} {tensor.shape}")

    # Save PyTorch model
    torch.save(pytorch_state_dict, pytorch_output_path)
    if verbose:
        logger.info(f"Saved PyTorch model to {pytorch_output_path}")

    return pytorch_state_dict


def _convert_tf_name_to_pytorch(tf_name):
    """Convert TensorFlow variable name to PyTorch parameter name.

    Parameters
    ----------
    tf_name : str
        TensorFlow variable name.

    Returns
    -------
    str
        PyTorch parameter name.
    """
    # Handle different naming conventions
    name = tf_name.replace("_weights", ".weight")
    name = name.replace("_bias", ".bias")
    name = name.replace("W", ".weight")
    name = name.replace("b", ".bias")

    # Handle merge layer naming
    name = name.replace("_input_1_weights", ".weight_im")
    name = name.replace("_input_2_weights", ".weight_pose")

    return name


def load_tf_config(model_dir):
    """Load configuration from TensorFlow model directory.

    Parameters
    ----------
    model_dir : str
        Directory containing TensorFlow model.

    Returns
    -------
    dict
        Model configuration.
    """
    config_file = os.path.join(model_dir, "config.json")
    with open(config_file, "r") as f:
        config = json.load(f, object_pairs_hook=OrderedDict)
    return config


def load_tf_architecture(model_dir):
    """Load architecture from TensorFlow model directory.

    Parameters
    ----------
    model_dir : str
        Directory containing TensorFlow model.

    Returns
    -------
    dict
        Model architecture.
    """
    arch_file = os.path.join(model_dir, "architecture.json")
    if os.path.exists(arch_file):
        with open(arch_file, "r") as f:
            arch = json.load(f, object_pairs_hook=OrderedDict)
        return arch
    return None


class TFCheckpointConverter:
    """Converter class for loading TensorFlow GQ-CNN weights into PyTorch models.
    
    Parameters
    ----------
    tf_model_dir : str
        Directory containing TensorFlow model checkpoint.
    config : dict, optional
        Model configuration. If None, will try to load from model directory.
    
    Examples
    --------
    >>> from gqcnn_torch import get_gqcnn_model
    >>> from gqcnn_torch.utils import TFCheckpointConverter, load_tf_config
    >>> 
    >>> # Load config and create model
    >>> cfg = load_tf_config("/path/to/tf_model")
    >>> model = get_gqcnn_model("GQCNN-4.0-PJ", cfg)
    >>> 
    >>> # Convert and load weights
    >>> converter = TFCheckpointConverter("/path/to/tf_model", cfg)
    >>> converter.load_to_model(model)
    """
    
    def __init__(self, tf_model_dir, config=None):
        """Initialize converter with TensorFlow model directory."""
        self.tf_model_dir = tf_model_dir
        self.config = config if config is not None else load_tf_config(tf_model_dir)
        self._state_dict = None
        
    @property
    def state_dict(self):
        """Get converted PyTorch state dict (lazy loading)."""
        if self._state_dict is None:
            self._state_dict = self._convert_checkpoint()
        return self._state_dict
    
    def _convert_checkpoint(self):
        """Convert TensorFlow checkpoint to PyTorch state dict."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TensorFlow is required for weight conversion. "
                "Install with: pip install tensorflow"
            )
        
        logger.info(f"Loading TensorFlow checkpoint from {self.tf_model_dir}")
        
        # Try different checkpoint file patterns
        ckpt_file = None
        for pattern in ["model.ckpt", "model", ""]:
            path = os.path.join(self.tf_model_dir, pattern)
            if tf.train.checkpoint_exists(path):
                ckpt_file = path
                break
        
        if ckpt_file is None:
            # Try to find .index file
            for f in os.listdir(self.tf_model_dir):
                if f.endswith(".index"):
                    ckpt_file = os.path.join(self.tf_model_dir, f[:-6])
                    break
        
        if ckpt_file is None:
            raise FileNotFoundError(
                f"No TensorFlow checkpoint found in {self.tf_model_dir}"
            )
        
        reader = tf.train.NewCheckpointReader(ckpt_file)
        var_to_shape_map = reader.get_variable_to_shape_map()
        
        pytorch_state_dict = OrderedDict()
        
        for var_name in sorted(var_to_shape_map.keys()):
            # Skip optimizer variables
            if "Adam" in var_name or "beta" in var_name or "global_step" in var_name:
                continue
                
            tensor = reader.get_tensor(var_name)
            pytorch_name = self._convert_name(var_name)
            
            # Handle weight transposition
            if "conv" in var_name.lower() and len(tensor.shape) == 4:
                # TF: [H, W, C_in, C_out] -> PyTorch: [C_out, C_in, H, W]
                tensor = np.transpose(tensor, (3, 2, 0, 1))
            elif len(tensor.shape) == 2:
                # TF: [in, out] -> PyTorch: [out, in]
                tensor = np.transpose(tensor, (1, 0))
            
            pytorch_state_dict[pytorch_name] = torch.from_numpy(tensor.copy())
            logger.debug(f"Converted {var_name} -> {pytorch_name} {tensor.shape}")
        
        logger.info(f"Converted {len(pytorch_state_dict)} variables")
        return pytorch_state_dict
    
    def _convert_name(self, tf_name):
        """Convert TensorFlow variable name to PyTorch parameter name."""
        # Extract layer name and weight type
        parts = tf_name.split("/")
        
        # Handle different naming conventions
        if len(parts) >= 2:
            layer_name = parts[-2]
            weight_type = parts[-1]
        else:
            layer_name = ""
            weight_type = parts[0]
        
        # Convert weight type
        if weight_type in ("W", "weights", "kernel"):
            weight_type = "weight"
        elif weight_type in ("b", "biases", "bias"):
            weight_type = "bias"
        
        # Build PyTorch name
        if layer_name:
            return f"{layer_name}.{weight_type}"
        return weight_type
    
    def load_to_model(self, model, strict=False):
        """Load converted weights into a PyTorch model.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model to load weights into.
        strict : bool
            If True, raise error on missing/unexpected keys.
            
        Returns
        -------
        tuple
            (missing_keys, unexpected_keys) from load_state_dict.
        """
        state_dict = self.state_dict
        
        # Try to match keys with model
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        # Create mapping for mismatched names
        matched_state_dict = OrderedDict()
        unmatched_ckpt = []
        
        for ckpt_key, value in state_dict.items():
            if ckpt_key in model_keys:
                matched_state_dict[ckpt_key] = value
            else:
                # Try to find matching key
                matched = False
                for model_key in model_keys:
                    if ckpt_key.split(".")[-1] == model_key.split(".")[-1]:
                        # Match by suffix
                        if ckpt_key.split(".")[0] in model_key:
                            matched_state_dict[model_key] = value
                            matched = True
                            break
                if not matched:
                    unmatched_ckpt.append(ckpt_key)
        
        missing_keys = model_keys - set(matched_state_dict.keys())
        
        if unmatched_ckpt:
            logger.warning(f"Unmatched checkpoint keys: {unmatched_ckpt}")
        if missing_keys:
            logger.warning(f"Missing model keys: {missing_keys}")
        
        result = model.load_state_dict(matched_state_dict, strict=strict)
        logger.info(f"Loaded {len(matched_state_dict)} parameters into model")
        
        return result
    
    def save_pytorch(self, output_path):
        """Save converted weights to PyTorch file.
        
        Parameters
        ----------
        output_path : str
            Path to save PyTorch model file.
        """
        torch.save(self.state_dict, output_path)
        logger.info(f"Saved PyTorch weights to {output_path}")
