#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert TensorFlow GQCNN models to PyTorch .pt format.

Usage:
    python convert_tf_models.py --input_dir /path/to/gqcnn/models --output_dir /path/to/output
    
    # Convert all models from gqcnn/models to gqcnn_torch/models
    python convert_tf_models.py
"""
import os
import sys
import json
import argparse
import shutil
from collections import OrderedDict

import numpy as np
import torch


def find_checkpoint(model_dir):
    """Find TensorFlow checkpoint file in model directory."""
    # Look for checkpoint patterns
    for pattern in ["model.ckpt", "model"]:
        index_file = os.path.join(model_dir, f"{pattern}.index")
        if os.path.exists(index_file):
            return os.path.join(model_dir, pattern)
    
    # Look for any .index file
    for f in os.listdir(model_dir):
        if f.endswith(".index"):
            return os.path.join(model_dir, f[:-6])
    
    return None


def convert_tf_name_to_pytorch(tf_name):
    """Convert TensorFlow variable name to PyTorch parameter name."""
    parts = tf_name.split("/")
    
    # Skip scope prefixes like 'gqcnn/', 'im_stream/', etc.
    # Keep only the layer name and weight type
    while len(parts) > 2:
        parts = parts[1:]
    
    if len(parts) >= 2:
        layer_name = parts[-2]
        weight_type = parts[-1]
    elif len(parts) == 1:
        # Handle flat names like 'fc3_weights'
        name = parts[0]
        if "_weights" in name:
            layer_name = name.replace("_weights", "")
            weight_type = "weight"
        elif "_bias" in name:
            layer_name = name.replace("_bias", "")
            weight_type = "bias"
        else:
            return name
        return f"{layer_name}.{weight_type}"
    else:
        return tf_name
    
    # Normalize weight type - convert TF naming to PyTorch naming
    if "weights" in weight_type.lower() or weight_type == "W" or weight_type == "kernel":
        weight_type = "weight"
    elif "bias" in weight_type.lower() or weight_type == "b":
        weight_type = "bias"
    else:
        # Keep original if not recognized
        pass
    
    return f"{layer_name}.{weight_type}"


def convert_model(model_dir, output_dir, verbose=True):
    """Convert a single TensorFlow model to PyTorch format.
    
    Parameters
    ----------
    model_dir : str
        Path to TensorFlow model directory.
    output_dir : str
        Path to output directory for PyTorch model.
    verbose : bool
        Whether to print progress.
    
    Returns
    -------
    bool
        True if conversion was successful.
    """
    try:
        import tensorflow as tf
        # Use compat.v1 for TF1-style checkpoints
        tf_compat = tf.compat.v1
    except ImportError:
        print("ERROR: TensorFlow is required for conversion.")
        print("Install with: pip install tensorflow")
        return False
    
    model_name = os.path.basename(model_dir)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Converting: {model_name}")
        print(f"{'='*60}")
    
    # Find checkpoint
    ckpt_path = find_checkpoint(model_dir)
    if ckpt_path is None:
        print(f"  WARNING: No checkpoint found in {model_dir}, skipping...")
        return False
    
    if verbose:
        print(f"  Checkpoint: {ckpt_path}")
    
    # Load checkpoint reader using TF1 compat API
    try:
        reader = tf_compat.train.NewCheckpointReader(ckpt_path)
    except Exception as e:
        print(f"  ERROR: Failed to read checkpoint: {e}")
        return False
    
    var_to_shape_map = reader.get_variable_to_shape_map()
    
    if verbose:
        print(f"  Found {len(var_to_shape_map)} TensorFlow variables")
    
    # Convert weights
    pytorch_state_dict = OrderedDict()
    converted_count = 0
    skipped_count = 0
    
    for var_name in sorted(var_to_shape_map.keys()):
        # Skip optimizer and non-model variables
        skip_patterns = ["Adam", "beta", "global_step", "Momentum", "ExponentialMovingAverage"]
        if any(pattern in var_name for pattern in skip_patterns):
            skipped_count += 1
            continue
        
        tensor = reader.get_tensor(var_name)
        pytorch_name = convert_tf_name_to_pytorch(var_name)
        
        # Transpose weights as needed
        if len(tensor.shape) == 4:
            # Conv: TF [H, W, C_in, C_out] -> PyTorch [C_out, C_in, H, W]
            tensor = np.transpose(tensor, (3, 2, 0, 1))
        elif len(tensor.shape) == 2:
            # FC: TF [in, out] -> PyTorch [out, in]
            tensor = np.transpose(tensor, (1, 0))
        
        pytorch_state_dict[pytorch_name] = torch.from_numpy(tensor.copy())
        converted_count += 1
        
        if verbose:
            print(f"    {var_name} -> {pytorch_name} {list(tensor.shape)}")
    
    if verbose:
        print(f"  Converted {converted_count} variables, skipped {skipped_count}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save PyTorch model
    pt_path = os.path.join(output_dir, "model.pt")
    torch.save(pytorch_state_dict, pt_path)
    if verbose:
        print(f"  Saved: {pt_path}")
    
    # Copy config files
    for config_file in ["config.json", "architecture.json"]:
        src = os.path.join(model_dir, config_file)
        if os.path.exists(src):
            dst = os.path.join(output_dir, config_file)
            shutil.copy2(src, dst)
            if verbose:
                print(f"  Copied: {config_file}")
    
    # Copy normalization files
    for npy_file in os.listdir(model_dir):
        if npy_file.endswith(".npy"):
            src = os.path.join(model_dir, npy_file)
            dst = os.path.join(output_dir, npy_file)
            shutil.copy2(src, dst)
            if verbose:
                print(f"  Copied: {npy_file}")
    
    if verbose:
        print(f"  SUCCESS: Model saved to {output_dir}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert TensorFlow GQCNN models to PyTorch format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing TensorFlow models (default: ../../../gqcnn/models)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for PyTorch models (default: ./models)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Convert only a specific model by name"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output"
    )
    
    args = parser.parse_args()
    verbose = not args.quiet and args.verbose
    
    # Set default paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(script_dir)
    
    if args.input_dir is None:
        # Default to gqcnn/models relative to workspace
        args.input_dir = os.path.abspath(
            os.path.join(package_dir, "..", "..", "gqcnn", "models")
        )
    
    if args.output_dir is None:
        # Default to gqcnn_torch/models
        args.output_dir = os.path.join(package_dir, "models")
    
    if verbose:
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
    
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Find models to convert
    models_to_convert = []
    
    if args.model:
        # Convert specific model
        model_path = os.path.join(args.input_dir, args.model)
        if os.path.isdir(model_path):
            models_to_convert.append(args.model)
        else:
            print(f"ERROR: Model not found: {model_path}")
            sys.exit(1)
    else:
        # Find all model directories (those with config.json or checkpoint files)
        for item in os.listdir(args.input_dir):
            item_path = os.path.join(args.input_dir, item)
            if os.path.isdir(item_path):
                # Check if it's a model directory
                has_config = os.path.exists(os.path.join(item_path, "config.json"))
                has_ckpt = find_checkpoint(item_path) is not None
                if has_config or has_ckpt:
                    models_to_convert.append(item)
    
    if not models_to_convert:
        print("No models found to convert.")
        print("Make sure the input directory contains model folders with config.json or checkpoint files.")
        sys.exit(1)
    
    if verbose:
        print(f"\nFound {len(models_to_convert)} model(s) to convert:")
        for m in models_to_convert:
            print(f"  - {m}")
    
    # Convert each model
    success_count = 0
    fail_count = 0
    
    for model_name in models_to_convert:
        input_path = os.path.join(args.input_dir, model_name)
        output_path = os.path.join(args.output_dir, model_name)
        
        if convert_model(input_path, output_path, verbose=verbose):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("Conversion Summary")
    print(f"{'='*60}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Output: {args.output_dir}")
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
