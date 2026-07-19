"""
SNN replacement controller trainer for the NeuCoDe toolkit.

Provides SNNWindowDataset for windowed time-series batching and
SNNReplacementTrainer which trains HybridControlSNN using snnTorch,
fits a MinMaxScaler, and saves model weights and calibration statistics.
"""
from __future__ import annotations
import os
import random
import numpy as np
import pandas as pd
import json
from pathlib import Path

from .._torch_optional import (
    torch, DataLoader, random_split, Dataset, 
    torch_available, snntorch_available ,
    snn
)

from ..ui import get_progress_bar
from ..architectures import HybridControlSNN
from .collector import collect_snn_layer_stats

class SNNWindowDataset(Dataset):
    """
    Dataset class for SNN training with windowed time-series data.
    Each sample consists of a window of context features and targets.
    """
    def __init__(self, features, targets, valid_indices, window_size):
        """
        Dataset setup.

        :param features: Tensor of shape [N, 5] containing context features.
        :param targets: Tensor of shape [N, 1] containing target control efforts.
        :param valid_indices: List or array of starting indices for valid windows.
        :param window_size: Size of the time window for each sample.
        """
        self.features = features
        self.targets = targets
        self.valid_indices = valid_indices
        self.window_size = window_size

    def __len__(self):
        """
        Return the number of valid windowed samples in the dataset.

        :returns: Number of valid start indices.
        """
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Retrieve a single windowed sample.

        :param idx: Index into valid_indices.
        :returns: Tuple (norm_error_chunk, target_chunk, context_chunk) each of
            shape [window_size, 1], [window_size, 1], and [window_size, 5] respectively.
        """
        start = self.valid_indices[idx]
        end = start + self.window_size

        chunk = self.features[start:end] 
        target_chunk = self.targets[start:end]

        norm_error_chunk = chunk[:, -1].unsqueeze(1)
        context_chunk = chunk[:, :-1]

        return norm_error_chunk, target_chunk, context_chunk

class SNNReplacementTrainer():
    """
    Trainer class for Spiking Neural Network (SNN) Replacement Controller.
    Implements training logic specific to SNNs using snnTorch.
    """
    TRAIN_VAL_SPLIT = 0.8

    @property
    def _use_cuda(self) -> bool:
        """
        True when the active training device is a CUDA GPU.
        """
        return self.device and self.device.type == 'cuda'

    def __init__(self):
        """
        Initializes the SNN Replacement Trainer.
        """
        if not torch_available or not snntorch_available:
            raise ImportError(f"{self.__class__.__name__} requires PyTorch & snnTorch.")
        self.model = None
        self.device = None

    _CONTEXT_COLUMNS = [
        'setpoint', 'measurement', 'error', 'integral_error', 'derivative_error',
    ]

    @staticmethod
    def _compute_feature_clips(ctx_df, clip_overrides=None):
        """Compute per-feature clip bounds from data statistics.

        Features 0-2 (setpoint, measurement, error): observed min/max.
        Feature 3 (integral_error): p1/p99.
        Feature 4 (derivative_error): symmetric p99 of absolute value.

        Returns (clip_min, clip_max) each of shape (5,).
        """
        columns = SNNReplacementTrainer._CONTEXT_COLUMNS
        clip_min = np.zeros(len(columns), dtype=np.float64)
        clip_max = np.zeros(len(columns), dtype=np.float64)

        for i, col in enumerate(columns):
            if i <= 2:
                clip_min[i] = ctx_df[col].min()
                clip_max[i] = ctx_df[col].max()
            elif i == 3:
                clip_min[i] = float(ctx_df[col].quantile(0.01))
                clip_max[i] = float(ctx_df[col].quantile(0.99))
            elif i == 4:
                p99 = float(ctx_df[col].abs().quantile(0.99))
                clip_min[i] = -p99
                clip_max[i] = p99

        if clip_overrides:
            for idx, (lo, hi) in clip_overrides.items():
                clip_min[idx] = lo
                clip_max[idx] = hi

        return clip_min, clip_max

    def _setup_training(self, learning_rate: float, hidden_size: int = 256, architecture: str = "hybrid", population_size: int = 64, beta: float = 0.92, rate_zero_point: float = 0.0, max_rate_input: float = 1.0):
        """
        Prepare the model, device, and Adam optimizer.

        :param learning_rate: Learning rate for the Adam optimizer.
        :param hidden_size: Number of hidden neurons (default 256).
        :param architecture: 'hybrid' or 'population' (default 'hybrid').
        :param population_size: Population size for PopulationControlSNN (default 64).
        :param beta: LIF membrane decay factor per timestep (default 0.92).
        :returns: The configured Adam optimizer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"* Using device: {self.device}")

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if architecture == "population":
            from ..architectures import PopulationControlSNN
            model = PopulationControlSNN(hidden_size=hidden_size, population_size=population_size, beta=beta, max_rate_input=max_rate_input).to(self.device)
        elif architecture == "spike":
            from ..architectures import SpikeControlSNN
            model = SpikeControlSNN(hidden_size=hidden_size, population_size=population_size, beta=beta, rate_zero_point=rate_zero_point).to(self.device)
        else:
            model = HybridControlSNN(hidden_size=hidden_size, beta=beta).to(self.device)
            
        self.model = model

        if self.initial_model_path is not None:
            if not os.path.exists(self.initial_model_path):
                raise FileNotFoundError(f"Initial model path '{self.initial_model_path}' does not exist.")
            print(f"* Loading initial model weights from '{self.initial_model_path}'")
            state_dict = torch.load(self.initial_model_path, map_location=self.device)
            state_dict.pop("encoder.prev_spike_value", None)
            model.load_state_dict(state_dict, strict=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        return optimizer
    
    def _setup_dataloader(self, dataset: Dataset, batch_size: int, loader_args: dict, seed: int = 42):
        """
        Split dataset into train/val and set up DataLoaders.

        :param dataset: The full SNNWindowDataset to split.
        :param batch_size: Number of samples per batch.
        :param loader_args: Extra keyword arguments forwarded to DataLoader (e.g. pin_memory).
        :param seed: Random seed for the reproducible train/val split.
        """
        train_len = int(self.TRAIN_VAL_SPLIT * len(dataset))
        val_len = len(dataset) - train_len
        
        def worker_init_fn(worker_id):
            np.random.seed(seed + worker_id)
            random.seed(seed + worker_id)
        
        generator = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)
        
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, **loader_args)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, **loader_args)

    @staticmethod
    def _get_scaler_class():
        """
        Dynamically imports and returns the MinMaxScaler class.
        """
        try:
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler
        except ImportError:
            raise ImportError("scikit-learn required.")

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """
        Validate the model on the validation dataset.

        SNN state is reset before each batch to prevent gradient/state leakage
        across episode boundaries.

        :param val_loader: DataLoader for the validation dataset.
        :returns: Average MSE loss over all validation batches.
        """
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=self._use_cuda):
                for error, target, context in val_loader:
                    # Move to GPU
                    error = error.to(self.device, non_blocking=self._use_cuda)
                    target = target.to(self.device, non_blocking=self._use_cuda)
                    context = context.to(self.device, non_blocking=self._use_cuda)
                    
                    self.model.reset_states() # Critical for SNN
                    pred, _ = self.model(error, context)
                    # Convert SNN spike probability [0, 1] to [-0.5, 0.5] range for comparison
                    pred = pred - 0.5
                    
                    # Robust Shape Fixes
                    if target.dim() == 2: target = target.unsqueeze(2)
                    if pred.shape[0] != target.shape[0] and pred.shape[1] == target.shape[0]: 
                        pred = pred.transpose(0, 1)
                    
                    # Standard MSE for validation reporting
                    val_loss += ((pred - target)**2).mean().item()

        return  val_loss / len(val_loader)

    def train(
        self,
        dataset_path: str | Path = None,
        model_save_filename: str | Path = None,
        scaler_save_filename: str | Path = None,
        stats_save_filename: str | Path = None,
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        hidden_size: int = None,
        architecture: str = None,
        population_size: int = None,
        window_size: int = None,
        stride: int = None,
        target_scale: float = None,
        mse_weight: float = None,
        cache_prefix: str | None = None,
        seed: int = None,
        verbose: bool = True,
        initial_model_path: str | Path = None,
        frozen_scaler_path: str | Path = None,
        beta: float = None,
        max_rate_input: float = None,
        clip_overrides: dict | None = None,
        experiment=None,
        tag: str = None,
        dataset_tag: str = None,
    ):
        """
        Train the SNN replacement controller end-to-end.

        :param dataset_path: Path to the CSV dataset file.
        :param model_save_filename: Filename for the saved PyTorch model weights.
        :param scaler_save_filename: Filename for the saved scaler parameters (.npz).
        :param stats_save_filename: Filename for the saved SNN calibration statistics (.json).
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        :param learning_rate: Learning rate for the Adam optimizer.
        :param hidden_size: Number of hidden neurons in HybridControlSNN (default 256).
        :param window_size: Number of time steps per SNN input window.
        :param stride: Step size between consecutive windows when building the dataset.
        :param target_scale: Divisor applied to raw control effort targets before training.
        :param mse_weight: Amplitude-weighted MSE multiplier (larger errors penalised more).
        :param cache_prefix: Path prefix for caching pre-processed NumPy arrays.
        :param seed: Random seed for reproducibility.
        :param verbose: If True, print the full training history on completion.
        :param frozen_scaler_path: Deprecated. Use clip_overrides instead.
        :param beta: LIF membrane decay factor per timestep (default 0.92).
            Scale with sample rate: beta_new = beta_ref^(dt_ref/dt_new).
        :param clip_overrides: Per-feature clip bounds as ``{feature_index: (lo, hi)}``.
            Overrides auto-computed clips for specified features.  Example:
            ``{3: (-50, 50)}`` to set integral bounds explicitly.
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
            cfg = experiment.training_defaults.get("snn", {})

        epochs = epochs if epochs is not None else cfg.get("epochs", 15)
        batch_size = batch_size if batch_size is not None else cfg.get("batch_size", 2048)
        learning_rate = learning_rate if learning_rate is not None else cfg.get("learning_rate", 0.01)
        hidden_size = hidden_size if hidden_size is not None else cfg.get("hidden_size", 256)
        architecture = architecture if architecture is not None else cfg.get("architecture", "hybrid")
        population_size = population_size if population_size is not None else cfg.get("population_size", 64)
        window_size = window_size if window_size is not None else cfg.get("window_size", 100)
        stride = stride if stride is not None else cfg.get("stride", 20)
        target_scale = target_scale if target_scale is not None else cfg.get("target_scale", 10.0)
        mse_weight = mse_weight if mse_weight is not None else cfg.get("mse_weight", 10.0)
        seed = seed if seed is not None else cfg.get("seed", experiment.config.get("seed", 42) if experiment else 42)
        beta = beta if beta is not None else cfg.get("beta", 0.92)
        max_rate_input = max_rate_input if max_rate_input is not None else cfg.get("max_rate_input", 1.0)

        # Ensure reproducibility with specified seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.initial_model_path = initial_model_path

        if model_save_filename is None:
            model_save_filename = "snn_replacement_model.pth"
        if scaler_save_filename is None:
            scaler_save_filename = "snn_replacement_scaler.npz"
        if stats_save_filename is None:
            stats_save_filename = "snn_replacement_stats.json"

        MinMaxScaler = self._get_scaler_class()

        # Training History for stats
        training_history = {
            'train_loss_weighted': [],
            'train_loss_raw': [],
            'val_loss': [],
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'window_size': window_size,
            'stride': stride,
            'target_scale': target_scale,
            'mse_weight': mse_weight,
        }
        
        # Caching
        dataset_dir = os.path.dirname(os.path.abspath(dataset_path))
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]
        if cache_prefix is None: 
            cache_prefix = os.path.join(dataset_dir, base_name)
        
        # Ensure cache dir exists
        cache_dir = os.path.dirname(cache_prefix)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        cache_feat = f"{cache_prefix}_snn_features.npy"
        cache_targ = f"{cache_prefix}_snn_targets.npy"
        cache_idx  = f"{cache_prefix}_snn_indices_s{stride}.npy"

        cache_meta = f"{cache_prefix}_snn_meta.npz"
        cache_files_exist = (os.path.exists(cache_feat) and os.path.exists(cache_targ)
                             and os.path.exists(cache_idx))
        cache_valid = (cache_files_exist
                       and frozen_scaler_path is None
                       and clip_overrides is None)
        if cache_valid and os.path.exists(cache_meta):
            meta = np.load(cache_meta)
            cached_ts = float(meta['target_scale']) if 'target_scale' in meta else None
            if cached_ts != target_scale:
                print(f"* Cache target_scale={cached_ts} != requested {target_scale}, invalidating")
                cache_valid = False
        elif cache_valid:
            print("* Cache has no metadata, invalidating to ensure consistency")
            cache_valid = False
        if cache_valid:
            print("* Loading cached SNN tensors...")
            features = torch.from_numpy(np.load(cache_feat))
            targets = torch.from_numpy(np.load(cache_targ))
            valid_indices = np.load(cache_idx)
            if scaler_save_filename is not None and not os.path.exists(str(scaler_save_filename)):
                import shutil
                default_scaler = str(cache_feat).replace("_snn_features.npy", "_scaler.npz") \
                    .replace("_snn_features_refitted.npy", "_scaler.npz")
                fallback = str(Path(dataset_path).parent / "replacement_dataset_scaler.npz")
                for candidate in [default_scaler, fallback]:
                    if os.path.exists(candidate):
                        shutil.copy2(candidate, str(scaler_save_filename))
                        print(f"* Copied cached scaler to {scaler_save_filename}")
                        break
                else:
                    print(f"[warn] Could not find cached scaler to copy to {scaler_save_filename}. "
                          "Delete the feature cache to force a full reprocess.")
        else:
            print("* Processing SNN dataset (Vectorized Windowing)...")
            df = pd.read_csv(dataset_path)
            
            # Normalize Targets
            # This value is highly dependent on your plant & dataset!
            # We just scale to a output favorable for SNN training.
            print(f"* Normalizing targets by factor: {target_scale}")
            targets_np = (df['control_effort'].values / target_scale).astype(np.float32)
            
            # Prepare Context [Setpoint, Measurement, Error, Integral Error, Derivative Error]
            ctx_df = df[['setpoint', 'measurement', 'error', 'integral_error', 'derivative_error']].copy()

            if frozen_scaler_path is not None:
                import warnings
                warnings.warn(
                    "frozen_scaler_path is deprecated; use clip_overrides instead",
                    DeprecationWarning, stacklevel=2,
                )
                with np.load(frozen_scaler_path) as frozen:
                    clip_min_vals = frozen['data_min'].astype(np.float64).copy()
                    clip_max_vals = (frozen['data_min'] + frozen['data_scale']).astype(np.float64)
                    if 'deriv_clip' in frozen.files:
                        dc = float(frozen['deriv_clip'][0])
                        clip_min_vals[4] = -dc
                        clip_max_vals[4] = dc
                if clip_overrides:
                    for idx, (lo, hi) in clip_overrides.items():
                        clip_min_vals[idx] = lo
                        clip_max_vals[idx] = hi
            else:
                clip_min_vals, clip_max_vals = self._compute_feature_clips(
                    ctx_df, clip_overrides)

            columns = ['setpoint', 'measurement', 'error',
                       'integral_error', 'derivative_error']
            for i, col in enumerate(columns):
                ctx_df[col] = ctx_df[col].clip(clip_min_vals[i], clip_max_vals[i])
            print(f"* Per-feature clips: min={np.array2string(clip_min_vals, precision=2)}, "
                  f"max={np.array2string(clip_max_vals, precision=2)}")

            ctx_cols = ctx_df.values
            scaler = MinMaxScaler(feature_range=(0, 1))
            ctx_scaled = scaler.fit_transform(ctx_cols).astype(np.float32)
            data_min   = scaler.data_min_
            data_scale = scaler.data_range_

            deriv_clip = float(clip_max_vals[4])
            u_max_value = target_scale / 2.0
            np.savez(
                scaler_save_filename,
                data_min=data_min,
                data_scale=data_scale,
                target_scale=target_scale,
                max_rate_input=np.array([max_rate_input], dtype=np.float32),
                deriv_clip=np.array([deriv_clip], dtype=np.float32),
                clip_min=clip_min_vals.astype(np.float32),
                clip_max=clip_max_vals.astype(np.float32),
                u_min=np.array([-u_max_value], dtype=np.float32),
                u_max=np.array([ u_max_value], dtype=np.float32),
            )
            print(f"* Scaler parameters saved to {scaler_save_filename}")
            
            # This is the last context column
            error_np = df['error'].values.astype(np.float32)
            features_np = np.hstack([ctx_scaled, error_np.reshape(-1, 1)])
            
            # Calculate valid windows with stride and episode handling
            print(f"* Calculating valid windows (Stride={stride})...")
            if 'episode_id' not in df.columns:
                if 'time' in df.columns:
                    df['episode_id'] = (df['time'].diff() < 0).cumsum()
                else:
                    # No time column (e.g. DAgger aggregated CSVs): treat the
                    # entire dataset as a single continuous episode so windows
                    # span the full sequence without artificial breaks.
                    df['episode_id'] = 0
            
            ep_ids = df['episode_id'].values
            starts = ep_ids[:-window_size+1]
            ends = ep_ids[window_size-1:]
            
            # Find all valid windows and apply stride
            valid_mask = (starts == ends)
            all_valid_indices = np.where(valid_mask)[0]
            valid_indices = all_valid_indices[::stride]
            
            print(f"* Reduced dataset from {len(all_valid_indices)} to {len(valid_indices)} samples.")
            
            # Write Cache
            np.save(cache_feat, features_np)
            np.save(cache_targ, targets_np)
            np.save(cache_idx, valid_indices)
            np.savez(cache_meta, target_scale=np.array([target_scale], dtype=np.float32))
            
            features = torch.from_numpy(features_np)
            # We reshape targets here for easier indexing later
            targets = torch.from_numpy(targets_np).view(-1, 1)

        loader_args = {}
        if self._use_cuda:
            loader_args = {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True}

        full_dataset = SNNWindowDataset(features, targets, valid_indices, window_size)
        self._setup_dataloader(full_dataset, batch_size, loader_args, seed=seed)
        
        rate_zero_point = 0.0
        if architecture == "spike" and scaler_save_filename and os.path.exists(str(scaler_save_filename)):
            sc = np.load(str(scaler_save_filename))
            if sc['data_scale'][2] != 0:
                rate_zero_point = float((0.0 - sc['data_min'][2]) / sc['data_scale'][2])
        optimizer = self._setup_training(learning_rate, hidden_size=hidden_size, architecture=architecture, population_size=population_size, beta=beta, rate_zero_point=rate_zero_point, max_rate_input=max_rate_input)
        
        # Robust Scaler Init
        try:
            grad_scaler = torch.amp.GradScaler('cuda', enabled=self._use_cuda)
        except TypeError:
            grad_scaler = torch.cuda.amp.GradScaler(enabled=self._use_cuda)

        num_batches = len(self.train_loader)
        total_steps = epochs * num_batches
        print(f"* Starting SNN Training ({epochs} epochs, {num_batches} batches/epoch)...")
        
        # Training Loop
        with get_progress_bar(total=total_steps, description="Training SNN") as progress:
            for epoch in range(epochs):
                self.model.train()
                train_loss_accum = 0.0
                raw_train_loss_accum = 0.0
                
                for error, target, context in self.train_loader:
                    # Move to GPU
                    error = error.to(self.device, non_blocking=self._use_cuda)
                    target = target.to(self.device, non_blocking=self._use_cuda)
                    context = context.to(self.device, non_blocking=self._use_cuda)
                    
                    self.model.reset_states() 
                    optimizer.zero_grad()
                    
                    # Mixed Precision Forward
                    with torch.amp.autocast(device_type=self.device.type, enabled=self._use_cuda):
                        pred, _ = self.model(error, context)
                        # Convert SNN spike probability [0, 1] to [-0.5, 0.5] range for comparison
                        pred = pred - 0.5
                        
                        # A few robust shape fixes
                        if target.dim() == 2: target = target.unsqueeze(2)
                        if pred.shape[0] != target.shape[0] and pred.shape[1] == target.shape[0]:
                            pred = pred.transpose(0, 1)
                        if pred.shape != target.shape:
                            raise RuntimeError(f"Shape Mismatch! Pred: {pred.shape}, Target: {target.shape}")

                        # Weighted MSE Loss
                        weights = torch.abs(target) * mse_weight + 1.0
                        loss = ((pred - target)**2 * weights).mean()

                        # Calculate raw MSE (for reporting, not for backprop)
                        raw_mse = ((pred - target)**2).mean().item()
                    
                    # Backward
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    
                    train_loss_accum += loss.item()
                    raw_train_loss_accum += raw_mse
                    progress.update(1)
            
                avg_train_loss = train_loss_accum / num_batches
                avg_raw_train_loss = raw_train_loss_accum / num_batches
                avg_val_mse = self._validate_epoch(self.val_loader)

                training_history['train_loss_weighted'].append(avg_train_loss)
                training_history['train_loss_raw'].append(avg_raw_train_loss)
                training_history['val_loss'].append(avg_val_mse)
                
                status_text = f"Epoch {epoch+1}/{epochs} | Train(W): {avg_train_loss:.5f} | Train(Raw MSE): {avg_raw_train_loss:.5f} | Val(MSE): {avg_val_mse:.5f}"
                progress.update(0, status=status_text)

        # Save
        torch.save(self.model.state_dict(), model_save_filename)
        print(f"* SNN replacement training complete. Model saved to {model_save_filename}")

        # Print final training history
        if verbose:
            print("\n* Final Training History:")
            print(training_history)

        # Calibration run (this time we can hook into torch modules)
        print(f"* Running calibration pass for SNN stats collection...")
        self.model.load_state_dict(torch.load(model_save_filename, map_location=self.device))
        self.model.eval()

        snn_stats = {}
        hooks = []

        for name, layer in self.model.named_modules():
            if isinstance(layer, snn.Leaky):
                hook = layer.register_forward_hook(
                    collect_snn_layer_stats(snn_stats, name)
                )
                hooks.append(hook)

        with torch.no_grad():
            for error, target, context in self.val_loader:
                error = error.to(self.device)
                context = context.to(self.device)
                self.model.reset_states()
                self.model(error, context)
        
        for hook in hooks: hook.remove()

        json_output = {name: stat.to_dict() for name, stat in snn_stats.items()}
        stats_path = Path(stats_save_filename)
        stats_path.write_text(json.dumps(json_output, indent=4), encoding='utf-8')
        print(f"* SNN layer stats saved to {stats_path}")

        # Expose history so callers (e.g. pipeline scripts) can persist it without
        # relying on console output.
        self.training_history_ = training_history