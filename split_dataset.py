import numpy as np
import pandas as pd
from src.data_explorer import DatasetExplorer
from src.data_preprocessor import DataPreprocessor


def main():
    explorer = DatasetExplorer("data/data.csv")
    df = explorer.load_data()

    preprocessor = DataPreprocessor(random_state=42)
    X_train, y_train, X_val, y_val = preprocessor.fit_transform_split(df, val_size=0.2)

    preprocessor.save_params("data/normalization_params.json")

    print(f"Dataset divisé avec succès !")
    print(f"  - X_train shape : {X_train.shape}")
    print(f"  - y_train shape : {y_train.shape}")
    print(f"  - X_val shape   : {X_val.shape}")
    print(f"  - y_val shape   : {y_val.shape}")

    np.save("data/X_train", X_train)
    np.save("data/y_train", y_train)
    np.save("data/X_val", X_val)
    np.save("data/y_val", y_val)


if __name__ == "__main__":
    main()
