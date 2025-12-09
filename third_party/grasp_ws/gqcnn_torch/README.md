# GQ-CNN PyTorch

A PyTorch implementation of Grasp Quality Convolutional Neural Networks (GQ-CNN), 
based on the original TensorFlow implementation from Berkeley Autolab.

## Overview

This package provides a PyTorch port of the GQ-CNN architecture for predicting 
grasp quality from depth images. It is designed to be compatible with Python 3.11+
and modern PyTorch versions.

## Features

- PyTorch implementation of GQ-CNN and FC-GQ-CNN architectures
- Compatible with existing model configurations from the TensorFlow version
- Weight conversion utilities from TensorFlow checkpoints to PyTorch
- Training and inference pipelines
- ROS2 integration support via `gqcnn_interfaces` package

## Installation

```bash
# Create a conda environment with Python 3.11
conda create -n gqcnn_torch python=3.11
conda activate gqcnn_torch

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision

# Install the package
pip install -e .
```

## Usage

### Loading a Model

```python
from gqcnn_torch import get_gqcnn_model, get_fc_gqcnn_model

# Load a GQ-CNN model
GQCNN = get_gqcnn_model(backend="torch")
model = GQCNN.load("path/to/model_dir")

# Run inference
predictions = model.predict(depth_images, poses)
```

### Training

```python
from gqcnn_torch import get_gqcnn_trainer

Trainer = get_gqcnn_trainer(backend="torch")
trainer = Trainer(config)
trainer.train()
```

### Converting TensorFlow Weights

```python
from gqcnn_torch.utils import convert_tf_to_pytorch

convert_tf_to_pytorch(
    tf_model_dir="path/to/tf/model",
    pytorch_output_path="path/to/pytorch/model.pt"
)
```

## Architecture

The package structure mirrors the original gqcnn package:

```
gqcnn_torch/
├── model/           # Neural network models
├── training/        # Training utilities
├── grasping/        # Grasp planning and policies
├── utils/           # Utility functions
├── cfg/             # Configuration files
└── scripts/         # Command-line scripts
```

## Differences from TensorFlow Version

1. Uses PyTorch tensors and autograd instead of TensorFlow
2. GPU memory is managed automatically by PyTorch
3. Models can be saved/loaded using `torch.save`/`torch.load`
4. Training uses PyTorch DataLoader for efficient data loading

## Citation

If you use this code, please cite the original GQ-CNN paper:

```
@article{mahler2017dex,
  title={Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic 
         Point Clouds and Analytic Grasp Metrics},
  author={Mahler, Jeffrey and Liang, Jacky and Niyaz, Sherdil and 
          Laskey, Michael and Doan, Richard and Liu, Xinyu and 
          Ojea, Juan Aparicio and Goldberg, Ken},
  journal={arXiv preprint arXiv:1703.09312},
  year={2017}
}
```

## License

BSD License (same as original gqcnn package)
