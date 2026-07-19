"""
Supervised-learning-based PID tuner for the NeuCoDe toolkit.

Provides SupervisedTuner which trains a small MLP to map plant observation
vectors to optimal PID gains from a pre-collected dataset.
"""
import os
import numpy as np
import pandas as pd
from typing import List, Optional

from .base import BaseTuner
from ..plants import Plant
from ..controllers import PIDController
from ..ui import get_progress_bar

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
    torch_available = True
except ImportError:
    print("Warning: PyTorch not found. SupervisedTuner will not be available.")
    torch_available = False
    class nn:
        """
        Dummy replacement for torch.nn when PyTorch is unavailable.
        """
        Module = object
    torch = None

class _SupervisedANN(nn.Module):
    """
    (Internal) A simple, default MLP architecture for the supervised tuner.
    It will be dynamically sized based on the dataset.
    """
    def __init__(self, in_features, out_features):
        """
        Build the default MLP.

        :param in_features: Number of plant observation features (input dimension).
        :param out_features: Number of PID gains to predict (output dimension).
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_features)
        )
    def forward(self, x):
        """
        Forward pass through the MLP.

        :param x: Input feature tensor of shape [batch, in_features].
        :returns: Predicted gains tensor of shape [batch, out_features].
        """
        return self.network(x)

class SupervisedTuner(BaseTuner):
    """
    A tuner that uses a supervised learning approach to predict optimal PID gains.
    """
    def __init__(self, model_arch: Optional[nn.Module] = None, device: str = "auto"):
        """
        Initialize the supervised tuner.

        :param model_arch: Optional custom nn.Module to use as the gain-prediction network.
            Defaults to a two-layer MLP sized dynamically from the dataset at train time.
        :param device: Torch device ('auto', 'cpu', or 'cuda').
        :raises ImportError: If PyTorch is not installed.
        """
        if torch is None:
            raise ImportError("PyTorch is required to use SupervisedTuner.")
            
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = model_arch.to(self.device) if model_arch else None
        self._is_trained = False
        self._in_features = 0
        self._out_features = 0
        self._label_column_names: List[str] = []

        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None 

        print(f"SupervisedTuner initialized on device '{self.device}'.")

    def train(self, 
              dataset_path: str, 
              epochs: int = 500,
              learning_rate: float = 1e-3, 
              save_path: str = None,
              validation_split: float = 0.2,
              early_stopping_patience: int = 20):
        """
        Train the supervised gain-prediction model.

        The CSV must contain columns prefixed with ``obs_`` (plant observation features)
        and columns containing ``_optimal`` (target PID gains).

        :param dataset_path: Path to the CSV dataset file.
        :param epochs: Maximum number of training epochs.
        :param learning_rate: Learning rate for the Adam optimizer.
        :param save_path: Optional path to save the best model checkpoint.
        :param validation_split: Fraction of data reserved for validation.
        :param early_stopping_patience: Epochs without validation improvement before stopping.
        """

        print("* Starting Supervised Training (Offline)")
        try:
            df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            print(f"* FATAL: Dataset file not found at {dataset_path}")
            return

        # this is what makes the tuner "plant-agnostic"
        # we dynamically find feature and label columns
        feature_cols = sorted([col for col in df.columns if col.startswith('obs_')])
        label_cols = sorted([col for col in df.columns if '_optimal' in col])
        self._in_features = len(feature_cols)
        self._out_features = len(label_cols)
        self._label_column_names = label_cols
        print(f"* Found {self._in_features} features: {feature_cols}")
        print(f"* Found {self._out_features} labels: {label_cols}")

        # If a user wants to provide a custom model architecture
        if self.model is None:
            print("* No model architecture provided, creating default MLP.")
            self.model = _SupervisedANN(self._in_features, self._out_features).to(self.device)

        X_df = df[feature_cols].astype(np.float32)
        Y_df = df[label_cols].astype(np.float32)

        # Normalization of inputs and outputs
        self._x_mean = X_df.mean().to_numpy(dtype=np.float32).copy()
        self._x_std  = X_df.std().to_numpy(dtype=np.float32).copy()
        self._x_std[self._x_std == 0.0] = 1.0
        self._y_mean = Y_df.mean().to_numpy(dtype=np.float32).copy()
        self._y_std  = Y_df.std().to_numpy(dtype=np.float32).copy()
        self._y_std[self._y_std == 0.0] = 1.0
        X_norm = (X_df.to_numpy() - self._x_mean) / self._x_std
        Y_norm = (Y_df.to_numpy() - self._y_mean) / self._y_std

        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_norm, dtype=torch.float32)

        # Split the dataset into training and validation sets
        dataset = TensorDataset(X_tensor, Y_tensor)
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Prepare for early stopping
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        
        with get_progress_bar("Training Supervised Model", total=epochs) as progress:
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                total_train_loss = 0
                for X_batch, Y_batch in train_loader:
                    X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                    optimizer.zero_grad()
                    Y_pred = self.model(X_batch)
                    loss = criterion(Y_pred, Y_batch)
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
                avg_train_loss = total_train_loss / len(train_loader)

                # Validation phase
                self.model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for X_batch, Y_batch in val_loader:
                        X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                        Y_pred = self.model(X_batch)
                        loss = criterion(Y_pred, Y_batch)
                        total_val_loss += loss.item()
                avg_val_loss = total_val_loss / len(val_loader)

                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    best_model_state = self.model.state_dict()
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= early_stopping_patience:
                    break

                # Our progress bar can handle those messages
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                
                progress.update()

        # Load the best model state
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        self.model.eval()
        self._is_trained = True
        print("*Supervised Training Complete")
        
        if save_path:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'in_features': self._in_features,
                'out_features': self._out_features,
                'label_column_names': self._label_column_names,
                'x_mean': self._x_mean,
                'x_std': self._x_std,
                'y_mean': self._y_mean,
                'y_std': self._y_std,
            }
            # check if path exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(checkpoint, save_path)
            print(f"*Best supervised model saved to {save_path}")

    def load(self, model_path: str = None):
        """
        Load a pre-trained supervised model checkpoint.

        :param model_path: Path to the saved checkpoint file produced by train().
        """
        print(f"* Loading pre-trained supervised model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self._in_features = checkpoint['in_features']
        self._out_features = checkpoint['out_features']
        self._label_column_names = checkpoint.get('label_column_names', [])
        self._x_mean = checkpoint.get('x_mean', None)
        self._x_std  = checkpoint.get('x_std', None)
        self._y_mean = checkpoint.get('y_mean', None)
        self._y_std  = checkpoint.get('y_std', None)

        if self.model is None:
            self.model = _SupervisedANN(self._in_features, self._out_features).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self._is_trained = True

    def tune(self, plant: Plant) -> PIDController:
        """
        Predict optimal PID gains for the given plant using the trained model.

        :param plant: A Plant instance whose get_observation_vector() is used as the model input.
        :returns: A PIDController with kp, ki, and kd gains predicted by the network.
        :raises RuntimeError: If the tuner has not been trained or loaded yet.
        """
        if not self._is_trained:
            raise RuntimeError("The supervised tuner must be trained or loaded first.")
        
        obs = plant.get_observation_vector()

        # We need to normalize inputs
        obs = (obs - self._x_mean) / self._x_std

        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            gains_norm = self.model(obs_tensor).cpu().numpy()[0]

        # We need to denormalize outputs
        gains = gains_norm * self._y_std + self._y_mean

        gains_dict = dict(zip(self._label_column_names, gains))
        
        return PIDController(
            kp=float(gains_dict.get('kp_optimal', 0.0)),
            ki=float(gains_dict.get('ki_optimal', 0.0)),
            kd=float(gains_dict.get('kd_optimal_industrial', 0.0))
        )