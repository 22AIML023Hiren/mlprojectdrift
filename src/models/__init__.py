import os
from pathlib import Path
from typing import Any
import joblib
import torch

from .baselines import LogisticRegressionBaseline, RandomForestBaseline
from .gbm import LightGBMModel
from .mlp import TabularMLP


def get_model(name: str, **kwargs) -> Any:
    """
    Factory function returning an initialized classifier instance.
    Supported: 'logistic_regression', 'random_forest', 'lightgbm', 'tabular_mlp'
    """
    name_clean = name.lower().replace("-", "_")
    if name_clean in ["logistic_regression", "lr", "logreg"]:
        return LogisticRegressionBaseline(**kwargs)
    elif name_clean in ["random_forest", "rf"]:
        return RandomForestBaseline(**kwargs)
    elif name_clean in ["lightgbm", "lgb", "gbm"]:
        return LightGBMModel(**kwargs)
    elif name_clean in ["tabular_mlp", "mlp", "neural_net"]:
        return TabularMLP(**kwargs)
    else:
        raise ValueError(f"Unknown model name '{name}'. Choose from: 'logistic_regression', 'random_forest', 'lightgbm', 'tabular_mlp'")


def save_model(model: Any, filepath: str) -> None:
    """
    Serializes model checkpoint to disk.
    Supports Scikit-learn/LightGBM (joblib) and PyTorch (torch.save).
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, TabularMLP):
        checkpoint = {
            "model_type": "tabular_mlp",
            "hidden_dim": model.hidden_dim,
            "dropout_rate": model.dropout_rate,
            "state_dict": model.net.state_dict()
        }
        torch.save(checkpoint, path)
    else:
        joblib.dump(model, path)
    print(f"[ModelRegistry] Saved model checkpoint to {path}")


def load_model(filepath: str) -> Any:
    """
    Loads serialized model checkpoint from disk.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    if path.suffix == ".pt":
        checkpoint = torch.load(path, map_location="cpu")
        model = TabularMLP(
            hidden_dim=checkpoint["hidden_dim"],
            dropout_rate=checkpoint["dropout_rate"]
        )
        # Note: net needs to be initialized with input_dim before loading state_dict
        # We store input dimension or lazy load
        first_layer_weight = checkpoint["state_dict"]["fc1.weight"]
        input_dim = first_layer_weight.shape[1]
        from .mlp import TabularMLPNet
        model.net = TabularMLPNet(input_dim, model.hidden_dim, model.dropout_rate)
        model.net.load_state_dict(checkpoint["state_dict"])
        model.net.eval()
        return model
    else:
        return joblib.load(path)


__all__ = [
    "LogisticRegressionBaseline",
    "RandomForestBaseline",
    "LightGBMModel",
    "TabularMLP",
    "get_model",
    "save_model",
    "load_model"
]
