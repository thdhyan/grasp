# -*- coding: utf-8 -*-
"""
Training statistics logger for gqcnn_torch package.
"""
import logging
import os
import numpy as np


class TrainStatsLogger:
    """Logger for training statistics."""

    def __init__(self, output_dir):
        """
        Parameters
        ----------
        output_dir : str
            Directory to save training statistics.
        """
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._train_losses = []
        self._train_errors = []
        self._val_losses = []
        self._val_errors = []
        self._learning_rates = []
        self._train_iters = []
        self._val_iters = []

        self._logger = logging.getLogger(self.__class__.__name__)

    def log(self, train_loss, train_error, val_loss=None, val_error=None,
            learning_rate=None, iteration=None, is_val=False):
        """Log training statistics.

        Parameters
        ----------
        train_loss : float
            Training loss.
        train_error : float
            Training error.
        val_loss : float, optional
            Validation loss.
        val_error : float, optional
            Validation error.
        learning_rate : float, optional
            Current learning rate.
        iteration : int, optional
            Current iteration.
        is_val : bool
            Whether this is a validation step.
        """
        if not is_val:
            self._train_losses.append(train_loss)
            self._train_errors.append(train_error)
            if iteration is not None:
                self._train_iters.append(iteration)
            if learning_rate is not None:
                self._learning_rates.append(learning_rate)
        else:
            if val_loss is not None:
                self._val_losses.append(val_loss)
            if val_error is not None:
                self._val_errors.append(val_error)
            if iteration is not None:
                self._val_iters.append(iteration)

    def save(self):
        """Save training statistics to files."""
        np.save(os.path.join(self._output_dir, "train_losses.npy"),
                np.array(self._train_losses))
        np.save(os.path.join(self._output_dir, "train_errors.npy"),
                np.array(self._train_errors))
        np.save(os.path.join(self._output_dir, "val_losses.npy"),
                np.array(self._val_losses))
        np.save(os.path.join(self._output_dir, "val_errors.npy"),
                np.array(self._val_errors))
        np.save(os.path.join(self._output_dir, "learning_rates.npy"),
                np.array(self._learning_rates))
        np.save(os.path.join(self._output_dir, "train_eval_iters.npy"),
                np.array(self._train_iters))
        np.save(os.path.join(self._output_dir, "val_eval_iters.npy"),
                np.array(self._val_iters))

        self._logger.info(f"Saved training statistics to {self._output_dir}")
