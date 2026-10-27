from .baselines import LogisticRegressionBaseline, RandomForestBaseline
from .gbm import LightGBMModel
from .mlp import TabularMLP

def get_model(name: str, **kwargs):
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

__all__ = [
    "LogisticRegressionBaseline",
    "RandomForestBaseline",
    "LightGBMModel",
    "TabularMLP",
    "get_model"
]
