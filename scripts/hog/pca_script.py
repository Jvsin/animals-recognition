#TODO: ... WKLEIĆ KOD Z BOTA TO KAŻDY POTRAFI 

"""
PCA Script for HOG Features
Loads HOG features and applies PCA dimensionality reduction.
"""

import numpy as np
from pathlib import Path
import argparse
import pickle
import sys

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Bierzemy tylko N_COMPONENTS, żeby była spójność z całym projektem
try:
    from scripts.pca import N_COMPONENTS
except Exception as e:
    raise ImportError(
        f"Could not import 'pca' module from sys.path: {e}"
    )

# =====================================================================
# LOAD HOG FEATURES
# =====================================================================
def load_hog_features(features_path):
    """
    Load HOG features from NPZ file.
    """
    print(f"Loading HOG features from {features_path}...\n")

    data = np.load(features_path)
    X, y, filenames = [], [], []

    for key in data.files:
        class_name = key.split("/")[0]
        filename = key.split("/")[1]
        feats = data[key]

        X.append(feats)
        y.append(class_name)
        filenames.append(filename)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print(f"Loaded {len(X)} samples with {X.shape[1]} features each")
    print("Classes:", np.unique(y))
    print("Samples per class:")
    for label in np.unique(y):
        count = np.sum(y == label)
        print(f"  {label}: {count}")

    return X, y, filenames


# =====================================================================
# SAVE PCA RESULTS
# =====================================================================
def save_pca_results(X_reduced, y, filenames, pca_model, output_dir):
    """
    Save PCA-reduced features and PCA model + info.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # zapis z powrotem do słownika label/filename → vec
    features_dict = {}
    for feats, label, filename in zip(X_reduced, y, filenames):
        key = f"{label}/{filename}"
        features_dict[key] = feats

    features_file = output_path / "hog_pca_features.npz"
    np.savez(features_file, **features_dict)
    print(f"\nSaved PCA features to {features_file}")

    pca_file = output_path / "hog_pca_model.pkl"
    with open(pca_file, "wb") as f:
        pickle.dump(pca_model, f)
    print(f"Saved PCA model to {pca_file}")

    info_file = output_path / "hog_pca_info.txt"
    with open(info_file, "w") as f:
        f.write(f"Number of samples: {len(X_reduced)}\n")
        f.write(f"Original feature dimensions: {pca_model.n_features_in_}\n")
        f.write(f"Reduced feature dimensions: {X_reduced.shape[1]}\n")
        f.write(f"Requested PCA components: {N_COMPONENTS}\n")
        f.write(f"Used PCA components: {pca_model.n_components_}\n")
        f.write(
            f"Explained variance ratio sum: {pca_model.explained_variance_ratio_.sum():.4f}\n"
        )
        f.write("\nClasses: " + ", ".join(map(str, np.unique(y))) + "\n")
        f.write("\nSamples per class:\n")
        for label in np.unique(y):
            count = np.sum(y == label)
            f.write(f"  {label}: {count}\n")
    print(f"Saved info to {info_file}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Apply PCA to HOG features")
    parser.add_argument(
        "--features-dir",
        type=str,
        default="output/hog/features",
        help="Directory containing HOG features (hog_features.npz)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/hog_pca",
        help="Directory to save PCA results",
    )

    args = parser.parse_args()

    features_file = Path(args.features_dir) / "hog_features.npz"
    if not features_file.exists():
        print(f"Error: Features file not found: {features_file}")
        return

    X, y, filenames = load_hog_features(features_file)

    print(f"\nOriginal feature dimensions: {X.shape[1]}")
    print(f"Applying PCA with up to {N_COMPONENTS} components...\n")

    # Skalowanie jak w ORB/SIFT
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Bezpieczeństwo: nie więcej komponentów niż liczba próbek/cech
    n_components = min(N_COMPONENTS, X_scaled.shape[0], X_scaled.shape[1])

    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)

    print(f"Reduced feature dimensions: {X_reduced.shape[1]}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    save_pca_results(X_reduced, y, filenames, pca, args.output_dir)

    print("\n" + "=" * 50)
    print("HOG PCA processing complete!")
    print(f"Dimensionality reduced: {X.shape[1]} → {X_reduced.shape[1]}")
    print("=" * 50)


if __name__ == "__main__":
    main()
