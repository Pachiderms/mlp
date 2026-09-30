import numpy as np


class DenseLayer:

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "sigmoid",
        initializer: str = "he",
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation

        if initializer == "he":
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        else:
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(1.0 / input_dim)

        self.b = np.zeros((1, output_dim))

        self.A_prev = None
        self.Z = None
        self.A = None

        self.dW = None
        self.db = None

    def _apply_activation(self, Z: np.ndarray) -> np.ndarray:
        if self.activation_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
        elif self.activation_name == "relu":
            return np.maximum(0, Z)
        elif self.activation_name == "softmax":
            # Stabilité numérique
            exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        else:
            raise ValueError(f"Fonction d'activation inconnue : {self.activation_name}")

    def forward(self, A_prev: np.ndarray) -> np.ndarray:
        """Feedforward"""
        self.A_prev = A_prev
        self.Z = np.dot(A_prev, self.W) + self.b
        self.A = self._apply_activation(self.Z)
        return self.A
