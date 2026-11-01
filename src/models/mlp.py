"""
PyTorch Tabular Multi-Layer Perceptron (MLP) with Monte Carlo Dropout
for epistemic uncertainty quantification in deep tabular classification.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class TabularMLPNet(nn.Module):
    """
    Feedforward architecture with residual connections, layer normalization,
    and dropout layers kept active during inference for Monte Carlo uncertainty sampling.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.25):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc_out = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        residual = out
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out + residual)
        out = self.dropout2(out)

        logits = self.fc_out(out)
        return logits


class TabularMLP:
    """
    Scikit-learn compatible wrapper for TabularMLPNet with weighted cross-entropy
    and Monte Carlo Dropout epistemic uncertainty sampling.
    """
    def __init__(
        self,
        hidden_dim: int = 64,
        dropout_rate: float = 0.25,
        lr: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 15,
        random_state: int = 42
    ):
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TabularMLP":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        input_dim = X.shape[1]
        self.net = TabularMLPNet(input_dim, self.hidden_dim, self.dropout_rate).to(self.device)

        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Weighted loss to mitigate severe fraud imbalance
        n_pos = max((y == 1).sum(), 1)
        n_neg = max((y == 0).sum(), 1)
        pos_weight = min(float(n_neg / n_pos), 50.0)
        class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        self.net.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.net(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Standard deterministic evaluation (dropout disabled)."""
        self.net.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.net(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def mc_dropout_predict(
        self,
        X: pd.DataFrame,
        n_samples: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Monte Carlo Dropout: forces dropout layers active during test time
        across n_samples stochastic forward passes to estimate epistemic predictive variance.
        
        Returns:
            mean_probs: Average predicted probabilities [N, 2]
            variance: Predictive epistemic variance of positive class [N]
        """
        # Preserve original training/eval mode
        original_mode = self.net.training
        # Enable dropout during inference
        self.net.train()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)

        mc_preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.net(X_tensor)
                p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                mc_preds.append(p)

        mc_preds = np.array(mc_preds)  # shape: [n_samples, N]
        mean_p_fraud = np.mean(mc_preds, axis=0)
        variance = np.var(mc_preds, axis=0)

        mean_probs = np.column_stack([1.0 - mean_p_fraud, mean_p_fraud])
        # Restore original mode (eval if it was False)
        if not original_mode:
            self.net.eval()
        return mean_probs, variance
