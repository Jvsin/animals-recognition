"""
PCA Script for ORB Features
Loads ORB features and applies PCA dimensionality reduction.
"""

import numpy as np
from pathlib import Path
import argparse
import pickle
import sys

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import only N_COMPONENTS from global pca.py, żeby była spójność z HOG
try:
    from scripts.pca import N_COMPONENTS
except Exception as e:
    raise ImportError(
        f"Could not import 'pca' module from sys.path: {e}"
    )


# =====================================================================
# LOAD ORB FEATURES (.npz)
# =====================================================================
def load_orb_features(features_path):
    """
    Load ORB features from NPZ file.

    Args:
        features_path: path to orb_features.npz

    Returns:
        X: feature matrix (n_samples, n_features)
        y: label array
        filenames: list of image filenames
    """
    print(f"Loading ORB features from: {features_path}\n")

    data = np.load(features_path)
    X, y, filenames = [], [], []

    for key in data.files:
        class_name = key.split("/")[0]
        filename = key.split("/")[1]
        features = data[key]

        X.append(features)
        y.append(class_name)
        filenames.append(filename)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print(f"Loaded {len(X)} samples")
    print(f"Feature length per sample: {X.shape[1]}")
    print("\nClasses:", np.unique(y))
    print("Samples per class:")
    for cls in np.unique(y):
        print(f"  {cls}: {np.sum(y == cls)}")

    return X, y, filenames


# =====================================================================
# SAVE PCA RESULTS
# =====================================================================
def save_pca_results(X_red, y, filenames, pca_model, output_dir):
    """
    Save PCA results: reduced features, model, and info.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save reduced feature dictionary
    features_dict = {}
    for features, label, filename in zip(X_red, y, filenames):
        key = f"{label}/{filename}"
        features_dict[key] = features

    features_file = output_path / "orb_pca_features.npz"
    np.savez(features_file, **features_dict)
    print(f"\nSaved PCA features to: {features_file}")

    # Save PCA model
    model_file = output_path / "orb_pca_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(pca_model, f)
    print(f"Saved PCA model to: {model_file}")

    # Save PCA info
    info_file = output_path / "orb_pca_info.txt"
    with open(info_file, "w") as f:
        f.write(f"SAMPLES: {len(X_red)}\n")
        f.write(f"ORIGINAL DIM: {pca_model.n_features_in_}\n")
        f.write(f"REDUCED DIM: {X_red.shape[1]}\n")
        f.write(f"PCA COMPONENTS (requested): {N_COMPONENTS}\n")
        f.write(
            f"PCA COMPONENTS (used): {pca_model.n_components_}\n"
        )
        f.write(
            f"EXPLAINED VAR SUM: {pca_model.explained_variance_ratio_.sum():.4f}\n\n"
        )

        f.write("CLASS DISTRIBUTION:\n")
        for cls in np.unique(y):
            f.write(f"  {cls}: {np.sum(y == cls)} samples\n")

    print(f"Saved PCA info to: {info_file}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Apply PCA to ORB features")
    parser.add_argument(
        "--features-dir",
        type=str,
        default="output/orb/features",
        help="Folder containing orb_features.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/orb_pca",
        help="Output directory for PCA results",
    )

    args = parser.parse_args()

    features_file = Path(args.features_dir) / "orb_features.npz"

    if not features_file.exists():
        print(f"ERROR: ORB features file not found:\n{features_file}")
        return

    # Load ORB features
    X, y, filenames = load_orb_features(features_file)

    print(f"\nApplying PCA with up to {N_COMPONENTS} components...\n")

    # Standaryzacja
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Bezpieczna liczba komponentów: nie więcej niż liczba próbek i cech
    n_components = min(N_COMPONENTS, X_scaled.shape[0], X_scaled.shape[1])

    pca = PCA(n_components=n_components)
    X_red = pca.fit_transform(X_scaled)

    print(f"Reduced dimensions: {X_red.shape[1]}")
    print(f"Explained variance sum: {pca.explained_variance_ratio_.sum():.4f}")

    # Zapis wyników
    save_pca_results(X_red, y, filenames, pca, args.output_dir)

    print("\n" + "=" * 60)
    print("ORB PCA PROCESSING COMPLETE")
    print(f"Dimensionality reduced: {X.shape[1]} → {X_red.shape[1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
