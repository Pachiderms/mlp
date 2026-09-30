from src.data_explorer import DatasetExplorer


def main():
    explorer = DatasetExplorer("data/data.csv")
    explorer.load_data()
    explorer.summarize()

    # Graphiques
    explorer.plot_class_distribution()
    explorer.plot_feature_distributions(
        features=[
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
        ]
    )


if __name__ == "__main__":
    main()
