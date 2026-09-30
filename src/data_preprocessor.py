import json
import numpy as np
import pandas as pd


class DataPreprocessor:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.mean = None
        self.std = None
        self.feature_names = []

    def fit_transform_split(
        self,
        df: pd.DataFrame,
        val_size: float = 0.2,
        target_col: str = "diagnosis",
    ):
        """Prépare, normalise et sépare les données en Train et Validation Sets."""
        self.feature_names = [
            col for col in df.columns if col not in ["id", target_col]
        ]

        X = df[self.feature_names].to_numpy(dtype=np.float64)

        labels = df[target_col].to_numpy()
        y_onehot = np.zeros((len(labels), 2), dtype=np.float64)
        y_onehot[labels == "M"] = [1.0, 0.0]
        y_onehot[labels == "B"] = [0.0, 1.0]

        np.random.seed(self.random_state)
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        X_shuffled = X[indices]
        y_shuffled = y_onehot[indices]

        val_count = int(len(X) * val_size)
        x_train_raw, x_val_raw = X_shuffled[val_count:], X_shuffled[:val_count]
        y_train, y_val = y_shuffled[val_count:], y_shuffled[:val_count]

        self.mean = np.mean(x_train_raw, axis=0)
        self.std = np.std(x_train_raw, axis=0)

        self.std[self.std == 0] = 1e-8

        x_train = (x_train_raw - self.mean) / self.std
        x_val = (x_val_raw - self.mean) / self.std

        return x_train, y_train, x_val, y_val

    def save_params(self, filepath: str = "normalization_params.json"):
        """Sauvegarde les paramètres de normalisation (moyenne et écart-type) pour la prédiction."""
        params = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_names": self.feature_names,
        }
        with open(filepath, "w") as f:
            json.dump(params, f, indent=4)

    def load_params(self, filepath: str = "normalization_params.json"):
        """Charge les paramètres de normalisation."""
        with open(filepath, "r") as f:
            params = json.load(f)
        self.mean = np.array(params["mean"])
        self.std = np.array(params["std"])
        self.feature_names = params["feature_names"]

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """Applique la normalisation sur de nouvelles données (utilisé par le programme de prédiction)."""
        if self.mean is None or self.std is None:
            raise ValueError("Les paramètres de normalisation ne sont pas chargés.")
        return (X_raw - self.mean) / self.std
