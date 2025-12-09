# -*- coding: utf-8 -*-
"""
GQ-CNN Trainer implemented in PyTorch.
Based on the TensorFlow implementation from Berkeley Autolab.
"""
import json
import logging
import math
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ..model import GQCNNTorch
from ..utils import (
    TrainingMode,
    GQCNNFilenames,
    GQCNNTrainingStatus,
    TrainStatsLogger,
    InputDepthMode,
)

logger = logging.getLogger(__name__)


class GQCNNDataset(Dataset):
    """Dataset for GQ-CNN training."""

    def __init__(
        self,
        data_dir,
        split="train",
        im_height=32,
        im_width=32,
        im_channels=1,
        pose_dim=1,
        gripper_mode="parallel_jaw",
        input_depth_mode="pose_stream",
    ):
        """
        Parameters
        ----------
        data_dir : str
            Directory containing training data.
        split : str
            Data split ("train" or "val").
        im_height : int
            Image height.
        im_width : int
            Image width.
        im_channels : int
            Number of image channels.
        pose_dim : int
            Pose dimension.
        gripper_mode : str
            Gripper mode.
        input_depth_mode : str
            Input depth mode.
        """
        self.data_dir = data_dir
        self.split = split
        self.im_height = im_height
        self.im_width = im_width
        self.im_channels = im_channels
        self.pose_dim = pose_dim
        self.gripper_mode = gripper_mode
        self.input_depth_mode = input_depth_mode

        # Load data files
        self._load_data()

    def _load_data(self):
        """Load data from files."""
        split_dir = os.path.join(self.data_dir, self.split)

        # Check if data exists
        if not os.path.exists(split_dir):
            logger.warning(f"Data directory {split_dir} does not exist.")
            self.images = np.zeros((0, self.im_height, self.im_width, self.im_channels))
            self.poses = np.zeros((0, self.pose_dim))
            self.labels = np.zeros((0,))
            return

        # Load images, poses, and labels
        try:
            self.images = np.load(os.path.join(split_dir, "images.npy"))
            self.poses = np.load(os.path.join(split_dir, "poses.npy"))
            self.labels = np.load(os.path.join(split_dir, "labels.npy"))
        except FileNotFoundError:
            logger.warning(f"Data files not found in {split_dir}. Creating empty dataset.")
            self.images = np.zeros((0, self.im_height, self.im_width, self.im_channels))
            self.poses = np.zeros((0, self.pose_dim))
            self.labels = np.zeros((0,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        pose = self.poses[idx]
        label = self.labels[idx]

        # Convert to tensors
        # [H, W, C] -> [C, H, W]
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))
        pose = torch.from_numpy(pose.astype(np.float32))
        label = torch.tensor(label, dtype=torch.long)

        return image, pose, label


class GQCNNTrainerTorch:
    """GQ-CNN Trainer implemented in PyTorch."""

    def __init__(self, config, gqcnn=None, output_dir=None):
        """
        Parameters
        ----------
        config : dict
            Training configuration.
        gqcnn : GQCNNTorch, optional
            Pre-initialized GQ-CNN model.
        output_dir : str, optional
            Output directory for saving models and logs.
        """
        self._config = config
        self._parse_config()

        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "models", "gqcnn_torch")
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)

        # Initialize model
        if gqcnn is not None:
            self._gqcnn = gqcnn
        else:
            self._gqcnn = GQCNNTorch(self._gqcnn_config, verbose=True)

        # Setup device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._gqcnn.to(self._device)

        # Initialize optimizer
        self._setup_optimizer()

        # Initialize loss function
        if self._training_mode == TrainingMode.CLASSIFICATION:
            self._loss_fn = nn.CrossEntropyLoss()
        else:
            self._loss_fn = nn.MSELoss()

        # Training status
        self._status = GQCNNTrainingStatus.NOT_STARTED

        # Stats logger
        self._stats_logger = TrainStatsLogger(self._output_dir)

    def _parse_config(self):
        """Parse training configuration."""
        self._data_dir = self._config.get("data_dir", "data")
        self._training_mode = self._config.get("training_mode", TrainingMode.CLASSIFICATION)
        
        # GQ-CNN config
        self._gqcnn_config = self._config.get("gqcnn", {})
        
        # Training params
        train_config = self._config.get("training", {})
        self._batch_size = train_config.get("batch_size", 64)
        self._num_epochs = train_config.get("num_epochs", 100)
        self._learning_rate = train_config.get("learning_rate", 0.001)
        self._lr_decay_rate = train_config.get("lr_decay_rate", 0.95)
        self._lr_decay_step = train_config.get("lr_decay_step", 10000)
        self._momentum = train_config.get("momentum", 0.9)
        self._weight_decay = train_config.get("weight_decay", 0.0005)
        self._drop_rate = train_config.get("drop_rate", 0.0)
        
        # Validation params
        self._val_frequency = train_config.get("val_frequency", 1000)
        self._save_frequency = train_config.get("save_frequency", 5000)
        self._log_frequency = train_config.get("log_frequency", 100)

        # Early stopping
        self._early_stopping_patience = train_config.get("early_stopping_patience", 10)

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        self._optimizer = optim.Adam(
            self._gqcnn.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )

        self._scheduler = optim.lr_scheduler.StepLR(
            self._optimizer,
            step_size=self._lr_decay_step,
            gamma=self._lr_decay_rate,
        )

    def _setup_data_loaders(self):
        """Setup training and validation data loaders."""
        train_dataset = GQCNNDataset(
            self._data_dir,
            split="train",
            im_height=self._gqcnn.im_height,
            im_width=self._gqcnn.im_width,
            im_channels=self._gqcnn.num_channels,
            pose_dim=self._gqcnn.pose_dim,
            gripper_mode=self._gqcnn.gripper_mode,
            input_depth_mode=self._gqcnn.input_depth_mode,
        )

        val_dataset = GQCNNDataset(
            self._data_dir,
            split="val",
            im_height=self._gqcnn.im_height,
            im_width=self._gqcnn.im_width,
            im_channels=self._gqcnn.num_channels,
            pose_dim=self._gqcnn.pose_dim,
            gripper_mode=self._gqcnn.gripper_mode,
            input_depth_mode=self._gqcnn.input_depth_mode,
        )

        self._train_loader = DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self._val_loader = DataLoader(
            val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def train(self):
        """Run training loop."""
        logger.info("Starting training...")
        self._status = GQCNNTrainingStatus.SETTING_UP

        self._setup_data_loaders()

        self._status = GQCNNTrainingStatus.TRAINING

        best_val_loss = float("inf")
        patience_counter = 0
        global_step = 0

        for epoch in range(self._num_epochs):
            epoch_start_time = time.time()

            # Training
            self._gqcnn.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (images, poses, labels) in enumerate(self._train_loader):
                images = images.to(self._device)
                poses = poses.to(self._device)
                labels = labels.to(self._device)

                # Forward pass
                self._optimizer.zero_grad()
                outputs = self._gqcnn(images, poses, drop_rate=self._drop_rate)

                # Compute loss
                loss = self._loss_fn(outputs, labels)

                # Backward pass
                loss.backward()
                self._optimizer.step()

                # Update stats
                train_loss += loss.item()
                if self._training_mode == TrainingMode.CLASSIFICATION:
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                global_step += 1

                # Log
                if global_step % self._log_frequency == 0:
                    avg_loss = train_loss / (batch_idx + 1)
                    if self._training_mode == TrainingMode.CLASSIFICATION:
                        train_acc = train_correct / train_total
                        logger.info(
                            f"Epoch {epoch+1}/{self._num_epochs}, "
                            f"Step {global_step}, "
                            f"Loss: {avg_loss:.4f}, "
                            f"Accuracy: {train_acc:.4f}"
                        )
                    else:
                        logger.info(
                            f"Epoch {epoch+1}/{self._num_epochs}, "
                            f"Step {global_step}, "
                            f"Loss: {avg_loss:.4f}"
                        )

                # Validation
                if global_step % self._val_frequency == 0:
                    val_loss, val_acc = self._validate()
                    logger.info(
                        f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}"
                    )

                    self._stats_logger.log(
                        train_loss=avg_loss,
                        train_error=1 - train_acc if self._training_mode == TrainingMode.CLASSIFICATION else avg_loss,
                        val_loss=val_loss,
                        val_error=1 - val_acc,
                        learning_rate=self._optimizer.param_groups[0]["lr"],
                        iteration=global_step,
                        is_val=True,
                    )

                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        self._save_model("best_model.pt")
                    else:
                        patience_counter += 1
                        if patience_counter >= self._early_stopping_patience:
                            logger.info(
                                f"Early stopping triggered after {patience_counter} validations without improvement."
                            )
                            break

                # Save model
                if global_step % self._save_frequency == 0:
                    self._save_model(f"model_step_{global_step}.pt")

                self._scheduler.step()

            # End of epoch
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

            if patience_counter >= self._early_stopping_patience:
                break

        # Save final model
        self._save_model(GQCNNFilenames.FINAL_MODEL_PYTORCH)
        self._save_config()
        self._stats_logger.save()

        logger.info("Training complete.")

    def _validate(self):
        """Run validation."""
        self._gqcnn.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, poses, labels in self._val_loader:
                images = images.to(self._device)
                poses = poses.to(self._device)
                labels = labels.to(self._device)

                outputs = self._gqcnn(images, poses)
                loss = self._loss_fn(outputs, labels)

                val_loss += loss.item()
                if self._training_mode == TrainingMode.CLASSIFICATION:
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

        avg_loss = val_loss / len(self._val_loader) if len(self._val_loader) > 0 else 0
        acc = val_correct / val_total if val_total > 0 else 0

        self._gqcnn.train()
        return avg_loss, acc

    def _save_model(self, filename):
        """Save model weights."""
        path = os.path.join(self._output_dir, filename)
        torch.save(self._gqcnn.state_dict(), path)
        logger.info(f"Saved model to {path}")

    def _save_config(self):
        """Save training configuration."""
        config_path = os.path.join(self._output_dir, GQCNNFilenames.SAVED_CFG)
        with open(config_path, "w") as f:
            json.dump(self._config, f, indent=2)

        arch_path = os.path.join(self._output_dir, GQCNNFilenames.SAVED_ARCH)
        with open(arch_path, "w") as f:
            json.dump(self._gqcnn._architecture, f, indent=2)

    @property
    def status(self):
        return self._status
