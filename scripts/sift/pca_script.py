"""
PCA Script for SIFT Features
Loads SIFT features and applies PCA dimensionality reduction.
"""

import numpy as np
from pathlib import Path
import argparse
import pickle
import sys

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import N_COMPONENTS from global pca.py (same value as for ORB/HOG)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pca import N_COMPONENTS


# =====================================================================
# LOAD SIFT FEATURES (.npz)
# =====================================================================
def load_sift_features(features_path):
    print(f"Loading SIFT features from: {features_path}\n")

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
# SAVE RESULTS
# =====================================================================
def save_pca_results(X_red, y, filenames, pca_model, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    features_dict = {}
    for feat, cls, fname in zip(X_red, y, filenames):
        key = f"{cls}/{fname}"
        features_dict[key] = feat

    features_file = out / "sift_pca_features.npz"
    np.savez(features_file, **features_dict)
    print(f"\nSaved PCA features to: {features_file}")

    model_file = out / "sift_pca_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(pca_model, f)
    print(f"Saved PCA model to: {model_file}")

    info_file = out / "sift_pca_info.txt"
    with open(info_file, "w") as f:
        f.write(f"SAMPLES: {len(X_red)}\n")
        f.write(f"ORIGINAL DIM: {pca_model.n_features_in_}\n")
        f.write(f"REDUCED DIM: {X_red.shape[1]}\n")
        f.write(f"REQUESTED COMPONENTS: {N_COMPONENTS}\n")
        f.write(f"USED COMPONENTS: {pca_model.n_components_}\n")
        f.write(f"EXPLAINED VARIANCE SUM: {pca_model.explained_variance_ratio_.sum():.4f}\n\n")

        f.write("CLASS DISTRIBUTION:\n")
        for cls in np.unique(y):
            f.write(f"  {cls}: {np.sum(y == cls)} samples\n")

    print(f"Saved PCA info to: {info_file}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Apply PCA to SIFT features")
    parser.add_argument(
        "--features-dir",
        type=str,
        default="output/sift/features",
        help="Directory containing sift_features.npz"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/sift_pca",
        help="Directory to save PCA output"
    )

    args = parser.parse_args()

    features_file = Path(args.features_dir) / "sift_features.npz"
    if not features_file.exists():
        print(f"ERROR: SIFT features file not found:\n{features_file}")
        return

    X, y, filenames = load_sift_features(features_file)

    print(f"\nApplying PCA with up to {N_COMPONENTS} components...\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(N_COMPONENTS, X_scaled.shape[0], X_scaled.shape[1])

    pca = PCA(n_components=n_components)
    X_red = pca.fit_transform(X_scaled)

    print(f"Reduced dimensions: {X_red.shape[1]}")
    print(f"Explained variance sum: {pca.explained_variance_ratio_.sum():.4f}")

    save_pca_results(X_red, y, filenames, pca, args.output_dir)

    print("\n" + "="*60)
    print("SIFT PCA PROCESSING COMPLETE")
    print(f"Dimensionality reduced: {X.shape[1]} → {X_red.shape[1]}")
    print("="*60)


if __name__ == "__main__":
    main()
