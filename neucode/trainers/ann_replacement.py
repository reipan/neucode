"""
ANN replacement controller trainer for the NeuCoDe toolkit.

Provides ANNReplacementTrainer which loads a PID-generated dataset, fits a
StandardScaler, trains a NoContextMLPArchitecture, and saves the model,
scaler, and calibration statistics for quantised export.
"""
from __future__ import annotations
import os
import random
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path

from .._torch_optional import torch, nn, DataLoader, random_split, TensorDataset, torch_available
from ..architectures import NoContextMLPArchitecture
from ..ui import get_progress_bar
from .collector import collect_linear_ranges

class ANNReplacementTrainer():
    """
    High-throughput trainer for the ANN-based PID replacement controller.
    """

    FEATURE_COLUMNS = ['setpoint', 'measurement', 'error', 'integral_error', 'derivative_error']
    TARGET_COLUMN = 'control_effort'
    TRAIN_VAL_SPLIT = 0.8

    @property
    def _use_cuda(self) -> bool:
        """
        True when the active training device is a CUDA GPU.
        """
        return self.device and self.device.type == 'cuda'

    def __init__(self, model_architecture: nn.Module = None, feature_columns: list = None):
        """
        Initializes the trainer.

        :param model_architecture: Optional custom PyTorch nn.Module to train.
            Defaults to NoContextMLPArchitecture sized for ``feature_columns``.
        :param feature_columns: List of CSV column names to use as input features.
            Defaults to the five standard PID signal columns.
        :raises ImportError: If PyTorch is not installed.
        :raises TypeError: If model_architecture is provided but is not an nn.Module.
        """
        if not torch_available:
            raise ImportError(f"{self.__class__.__name__} requires PyTorch to be installed.")
            
        self.model = None
        self.device = None
        self.feature_columns = feature_columns if feature_columns is not None else self.FEATURE_COLUMNS

        if model_architecture is None:
            # Default to the replacement-controller MLP over the configured feature columns.
            self.model_architecture = NoContextMLPArchitecture(input_size=len(self.feature_columns))
        else:
            if not isinstance(model_architecture, nn.Module):
                raise TypeError("model_architecture must be an instance of torch.nn.Module.")
            self.model_architecture = model_architecture

    def _setup_training(self, learning_rate: float):
        """
        Prepares model, device, criterion, and optimizer.

        :param learning_rate: Learning rate for the Adam optimizer.
        :returns: Tuple of (criterion, optimizer).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"* Using device: {self.device}")

        if self.device.type == "cuda":
            # Enable TF32 for faster FP32 math on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model = self.model_architecture.to(self.device)
        if self.initial_model_path is not None:
            if not os.path.exists(self.initial_model_path):
                raise FileNotFoundError(f"Initial model path '{self.initial_model_path}' does not exist.")
            print(f"* Loading initial model weights from '{self.initial_model_path}'")
            model.load_state_dict(torch.load(self.initial_model_path, map_location=self.device))
        
        # Compile model for speedup (PyTorch 2.0+)
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"* Warning: Could not compile model (torch.compile): {e}")

        self.model = model

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return criterion, optimizer
    
    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """
        Run one validation epoch and return the average loss.

        :param val_loader: DataLoader for the validation dataset.
        :param criterion: Loss function.
        :returns: Average validation loss over all batches.
        """
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for features, targets in val_loader:
                # Use non_blocking transfer if on CUDA
                features = features.to(self.device, non_blocking=self._use_cuda)
                targets = targets.to(self.device, non_blocking=self._use_cuda)

                # Faster and less memory-using validation using AMP
                with torch.amp.autocast(device_type='cuda', enabled=self._use_cuda):
                    outputs = self.model(features)
                    loss = criterion(outputs, targets.view(-1, 1))

                val_loss += loss.item()

        return val_loss / len(val_loader)
    
    @staticmethod
    def _get_scaler_class():
        """
        Dynamically import and return the StandardScaler class from scikit-learn.

        :raises ImportError: If scikit-learn is not installed.
        """
        try:
            from sklearn.preprocessing import StandardScaler
            return StandardScaler
        except ImportError:
            raise ImportError("This trainer requires scikit-learn for data scaling.")

    def train(
        self,
        dataset_path: str | Path = None,
        model_save_filename: str | Path = None,
        scaler_save_filename: str | Path = None,
        stats_save_filename: str | Path = None,
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        cache_prefix: str | None = None,
        use_amp: bool = None,
        num_workers: int = None,
        early_stopping_patience: int = None,
        early_stopping_delta: float = None,
        seed: int = None,
        verbose: bool = True,
        initial_model_path: str | Path = None,
        experiment=None,
        tag: str = None,
        dataset_tag: str = None,
    ):
        """
        Train the ANN replacement controller end-to-end.

        :param dataset_path: Path to the CSV dataset.
        :param model_save_filename: Filename for the saved PyTorch model weights.
        :param scaler_save_filename: Filename for the saved scaler parameters (.npz).
        :param stats_save_filename: Filename for the saved calibration statistics (.json).
        :param epochs: Number of training epochs.
        :param batch_size: Training batch size.
        :param learning_rate: Learning rate for the Adam optimizer.
        :param cache_prefix: Path prefix for caching pre-processed NumPy arrays.
        :param use_amp: Whether to use Automatic Mixed Precision on CUDA.
        :param num_workers: Number of DataLoader worker processes.
        :param early_stopping_patience: Epochs without improvement before stopping
            (0 = disabled).
        :param early_stopping_delta: Minimum validation loss improvement to reset the
            patience counter.
        :param seed: Random seed for reproducibility.
        :param verbose: If True, print the full training history on completion.
        :param experiment: Optional Experiment instance for path and config resolution.
        :param tag: Training tag within the experiment (required when experiment is passed).
        :param dataset_tag: Dataset tag if different from training tag.
        """
        # Resolve paths and defaults from experiment config
        cfg = {}
        if experiment is not None:
            if tag is None:
                raise ValueError("tag is required when passing experiment")
            dataset_path = dataset_path or experiment.get_dataset_path(dataset_tag or tag)
            model_save_filename = model_save_filename or experiment.get_model_path(tag)
            scaler_save_filename = scaler_save_filename or experiment.get_scaler_path(tag)
            stats_save_filename = stats_save_filename or experiment.get_stats_path(tag)
            cfg = experiment.training_defaults.get("ann", {})

        epochs = epochs if epochs is not None else cfg.get("epochs", 50)
        batch_size = batch_size if batch_size is not None else cfg.get("batch_size", 8192)
        learning_rate = learning_rate if learning_rate is not None else cfg.get("learning_rate", 0.001)
        use_amp = use_amp if use_amp is not None else cfg.get("use_amp", True)
        num_workers = num_workers if num_workers is not None else cfg.get("num_workers", 4)
        early_stopping_patience = early_stopping_patience if early_stopping_patience is not None else cfg.get("early_stopping_patience", 0)
        early_stopping_delta = early_stopping_delta if early_stopping_delta is not None else cfg.get("early_stopping_delta", 0.0001)
        seed = seed if seed is not None else cfg.get("seed", experiment.config.get("seed", 42) if experiment else 42)

        # Ensure reproducibility with specified seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.initial_model_path = initial_model_path

        training_start = time.time()
        if dataset_path is None:
            dataset_path = Path("artifacts/datasets/") / str(int(training_start))
        else:
            dataset_path = Path(dataset_path)

        if model_save_filename is None:
            model_save_filename = "ann_replacement_model.pth"
        if scaler_save_filename is None:
            scaler_save_filename = "ann_replacement_scaler.npz"
        if stats_save_filename is None:
            stats_save_filename = "ann_replacement_stats.json"

        StandardScaler = self._get_scaler_class()

        # Training History for stats
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_delta': early_stopping_delta,
            'training_time_seconds': None,
        }

        # Determine cache file locations
        dataset_dir = os.path.dirname(os.path.abspath(dataset_path))
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]

        if cache_prefix is None:
            cache_prefix = os.path.join(dataset_dir, base_name)

        # Ensure cache directory exists
        cache_dir = os.path.dirname(cache_prefix)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        num_features = len(self.feature_columns)
        features_cache = f"{cache_prefix}_features_{num_features}.npy"
        targets_cache = f"{cache_prefix}_targets_{num_features}.npy"

        # Decide whether to load from cache or parse CSV
        if os.path.exists(features_cache) and os.path.exists(targets_cache) and os.path.exists(scaler_save_filename):
            print("* Loading cached features/targets from .npy files...")
            features_scaled = np.load(features_cache)
            targets = np.load(targets_cache)

            # Load scaler params
            scaler_params = np.load(scaler_save_filename)
            print(
                f"* Loaded scaler params from '{scaler_save_filename}' "
                f"* (mean shape: {scaler_params['mean'].shape}, "
                f"* scale shape: {scaler_params['scale'].shape})"
            )
        else:
            print("* Loading and scaling data from CSV (first run fills cache)...")
            df = pd.read_csv(dataset_path, dtype=np.float32)

            # Clip derivative_error at 99th percentile before fitting the scaler.
            # Step-boundary spikes (delta sp/dt) dominate the std and collapse all normal
            # derivative values to ~0 after StandardScaler normalization.
            if 'derivative_error' in self.feature_columns:
                clip_val = float(df['derivative_error'].abs().quantile(0.99))
                df['derivative_error'] = df['derivative_error'].clip(-clip_val, clip_val)
                print(f"* Clipped derivative_error at +/-{clip_val:.2f} (99th pct)")
            else:
                clip_val = float('nan')

            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(df[self.feature_columns])
            targets = df[self.TARGET_COLUMN].values

            # Save scaler params (include deriv_clip so MCU can apply same bound)
            np.savez(scaler_save_filename, mean=scaler.mean_, scale=scaler.scale_,
                     deriv_clip=np.array([clip_val], dtype=np.float32))
            print(f"* Scaler parameters saved to '{scaler_save_filename}'")

            # Cache NumPy arrays for fast reloads
            np.save(features_cache, features_scaled)
            np.save(targets_cache, targets)
            print(f"* Cached features to '{features_cache}' and targets to '{targets_cache}'")

        # Convert to torch tensors ONCE
        features_tensor = torch.from_numpy(features_scaled)
        targets_tensor = torch.from_numpy(targets).view(-1, 1)

        full_dataset = TensorDataset(features_tensor, targets_tensor)

        # Train/validation split
        train_size = int(self.TRAIN_VAL_SPLIT * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        
        criterion, optimizer = self._setup_training(learning_rate)
        use_cuda = self._use_cuda

        # Create worker_init_fn with seed captured in closure
        def worker_init_fn(worker_id):
            np.random.seed(seed + worker_id)
            random.seed(seed + worker_id)

        loader_args = {}
        if use_cuda:
            loader_args = {
                'num_workers': num_workers,
                'pin_memory': True,
                'persistent_workers': True,
                'worker_init_fn': worker_init_fn,
            }
        else:
            loader_args = {'worker_init_fn': worker_init_fn}

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **loader_args
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, **loader_args
        )

        num_train_batches = len(train_loader)
        total_steps = epochs * num_train_batches
        print(f"* Training for {epochs} epochs with {num_train_batches} batches each ({total_steps} total steps).")

        model = self.model
        use_amp_enabled = use_amp and use_cuda
        
        # Robust Scaler Init
        try:
            grad_scaler = torch.amp.GradScaler('cuda', enabled=use_amp_enabled)
        except TypeError:
            grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp_enabled)

        # Early stopping vars
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None

        with get_progress_bar(total=total_steps, description="Training ANN") as progress:
            for epoch in range(epochs):
                model.train()
                running_train_loss = 0.0

                for features_batch, targets_batch in train_loader:
                    if use_cuda:
                        features_batch = features_batch.to(self.device, non_blocking=True)
                        targets_batch = targets_batch.to(self.device, non_blocking=True)
                    else:
                        features_batch = features_batch.to(self.device)
                        targets_batch = targets_batch.to(self.device)

                    optimizer.zero_grad()

                    with torch.amp.autocast(device_type='cuda', enabled=use_amp_enabled):
                        outputs = model(features_batch)
                        loss = criterion(outputs, targets_batch)

                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    running_train_loss += loss.item()
                    progress.update(1)

                avg_val_loss = self._validate_epoch(val_loader, criterion)
                avg_train_loss = running_train_loss / num_train_batches
                status_text = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.5f}, Val Loss: {avg_val_loss:.5f}"
                training_history['train_loss'].append(avg_train_loss)
                training_history['val_loss'].append(avg_val_loss)

                # Early stopping
                if early_stopping_patience > 0:
                    if avg_val_loss < best_val_loss - early_stopping_delta:
                        best_val_loss = avg_val_loss
                        epochs_without_improvement = 0
                        best_model_state = model.state_dict()
                        status_text += f" (New best: {best_val_loss:.5f})"
                    else:
                        epochs_without_improvement += 1
                        status_text += f" (No improvement for {epochs_without_improvement} epochs)"

                    if epochs_without_improvement >= early_stopping_patience:
                        progress.update(0, status=status_text)
                        print(f"\n* Early stopping triggered after {epoch + 1} epochs.")
                        break
                
                progress.update(0, status=status_text)

        if best_model_state:
            model.load_state_dict(best_model_state)
            print("\n* Loaded best model state from early stopping.")

        # Output training complete history
        if verbose:
            print("\n* Training complete. Final training history:")
            print(training_history)

        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model

        # Use validation loader for calibration to collect linear layer ranges for quantization
        stats = collect_linear_ranges(model=model_to_save, calibration_loader=val_loader)
        stats_path = Path(dataset_dir) / stats_save_filename
        stats_path.write_text(json.dumps(stats, indent=4), encoding='utf-8')
        print(f"* Collected stats saved to '{stats_path}'.")

        torch.save(model_to_save.state_dict(), model_save_filename)
        print(f"* ANN replacement training complete. Model saved to '{model_save_filename}'.")

        # Expose history so callers (e.g. pipeline scripts) can persist it without
        # relying on console output.
        self.training_history_ = training_history