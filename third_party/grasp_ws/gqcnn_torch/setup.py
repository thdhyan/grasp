# -*- coding: utf-8 -*-
"""
GQ-CNN PyTorch Implementation
A PyTorch port of the original TensorFlow-based GQ-CNN package.

This package provides grasp quality convolutional neural networks
implemented in PyTorch, compatible with Python 3.11+.

Author: Based on Berkeley Autolab's gqcnn (https://github.com/BerkeleyAutomation/gqcnn)
"""
import os
from setuptools import setup, find_packages

# Read version from version file
version = "1.0.0"

# Read requirements
requirements = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "scikit-image>=0.20.0",
    "scikit-learn>=1.2.0",
    "opencv-python>=4.7.0",
    "pyyaml>=6.0",
    "matplotlib>=3.7.0",
    "autolab-core>=1.1.0",
]

setup(
    name="gqcnn_torch",
    version=version,
    description="PyTorch implementation of Grasp Quality Convolutional Neural Networks",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Based on Berkeley Autolab",
    author_email="",
    url="https://github.com/BerkeleyAutomation/gqcnn",
    license="BSD",
    packages=find_packages(),
    package_data={
        "gqcnn_torch": [
            "cfg/*.yaml",
            "cfg/examples/*.yaml",
        ]
    },
    install_requires=requirements,
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "gqcnn_torch_policy=gqcnn_torch.scripts.run_policy:main",
            "gqcnn_torch_train=gqcnn_torch.scripts.train:main",
        ],
    },
)
