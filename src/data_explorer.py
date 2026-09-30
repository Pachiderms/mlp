import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DatasetExplorer:

    def __init__(self, filepath: str):
        """Initialise l'explorateur avec le chemin du fichier CSV."""
        self.filepath = filepath
        self.df = None
        self.feature_names = []
        self.target_col = None

    def load_data(self) -> pd.DataFrame:
        """Charge le jeu de données CSV et identifie les colonnes."""
        self.df = pd.read_csv(self.filepath, header=None)

        if self.df.iloc[0, 1] in ["M", "B"]:
            columns = [
                "id",
                "diagnosis",
                "radius_mean",
                "texture_mean",
                "perimeter_mean",
                "area_mean",
                "smoothness_mean",
                "compactness_mean",
                "concavity_mean",
                "concave_points_mean",
                "symmetry_mean",
                "fractal_dimension_mean",
                "radius_se",
                "texture_se",
                "perimeter_se",
                "area_se",
                "smoothness_se",
                "compactness_se",
                "concavity_se",
                "concave_points_se",
                "symmetry_se",
                "fractal_dimension_se",
                "radius_worst",
                "texture_worst",
                "perimeter_worst",
                "area_worst",
                "smoothness_worst",
                "compactness_worst",
                "concavity_worst",
                "concave_points_worst",
                "symmetry_worst",
                "fractal_dimension_worst",
            ]
            self.df.columns = columns

        self.target_col = "diagnosis"

        self.feature_names = [
            col for col in self.df.columns if col not in ["id", self.target_col]
        ]
        return self.df

    def summarize(self) -> None:
        """Affiche un résumé statistique et vérifie la présence de valeurs manquantes."""
        if self.df is None:
            raise ValueError("Veuillez d'abord charger les données avec load_data().")

        print("=" * 50)
        print("Résumé du Dataset")
        print("=" * 50)
        print(f"Dimensions : {self.df.shape[0]} lignes, {self.df.shape[1]} colonnes")
        print(
            "\nValeurs manquantes par colonne :\n",
            self.df[self.feature_names].isnull().sum().sum(),
        )

        print("\nRépartition de la cible (Diagnosis) :")
        counts = self.df[self.target_col].value_counts()
        for label, count in counts.items():
            pct = (count / len(self.df)) * 100
            print(
                f"  - {label} ({'Malignant' if label=='M' else 'Benign'}): {count} ({pct:.2f}%)"
            )

        print("\nAperçu des statistiques descriptives :")
        print(
            self.df[self.feature_names[:5]]
            .describe()
            .T[["mean", "std", "min", "50%", "max"]]
        )

    def plot_class_distribution(self, save_path: str = None) -> None:
        """Affiche un graphique en barres de la répartition des classes."""
        plt.figure(figsize=(6, 4))

        counts = self.df[self.target_col].value_counts()

        plt.bar(counts.index, counts.values, color=["skyblue", "salmon"])
        plt.title("Répartition des Diagnostics (M vs B)")

        plt.xlabel("Diagnostic")
        plt.ylabel("Nombre d'échantillons")

        plt.grid(axis="y", linestyle="--", alpha=0.7)

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_feature_distributions(
        self, features: list = None, save_path: str = None
    ) -> None:
        """Affiche les histogrammes comparatifs entre M et B pour un sous-ensemble de caractéristiques."""
        if features is None:
            features = self.feature_names[:6]  # Par défaut, les 6 premières

        fig, axes = plt.subplots(len(features) // 2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, feat in enumerate(features):
            ax = axes[idx]
            for diag, color in zip(["B", "M"], ["blue", "red"]):
                subset = self.df[self.df[self.target_col] == diag]
                ax.hist(
                    subset[feat],
                    bins=20,
                    alpha=0.5,
                    label="Malignant" if diag == "M" else "Benign",
                    color=color,
                    density=True,
                )
            ax.set_title(feat)
            ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
