
"""
PCA Script for LBP Features
Loads LBP features and applies PCA dimensionality reduction.
"""

import numpy as np
from pathlib import Path
import argparse
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =====================================================================
# LOAD LBP FEATURES
# =====================================================================
def load_lbp_features(features_path):
    """
    Load LBP features from NPZ file.

    Args:
        features_path: Path to lbp_features.npz

    Returns:
        X: Feature matrix
        y: Class labels
        filenames: Image names
    """
    print(f"Loading LBP features from: {features_path}")

    data = np.load(features_path)
    X = []
    y = []
    filenames = []

    for key in data.files:
        class_name = key.split("/")[0]
        filename = key.split("/")[1]
        feat = data[key]

        X.append(feat)
        y.append(class_name)
        filenames.append(filename)

    X = np.array(X)
    y = np.array(y)

    print(f"\nLoaded {len(X)} samples")
    print(f"Feature length per sample: {X.shape[1]}")

    print(f"\nClasses: {np.unique(y)}")
    print("Samples per class:")
    for cls in np.unique(y):
        print(f"  {cls}: {np.sum(y == cls)}")

    return X, y, filenames


# =====================================================================
# SAVE PCA RESULTS
# =====================================================================
def save_pca_results(Xred, y, filenames, pca, output_dir, original_dim):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save NPZ with reduced features
    features_dict = {}
    for features, label, filename in zip(Xred, y, filenames):
        key = f"{label}/{filename}"
        features_dict[key] = features

    npz_path = output_dir / "lbp_pca_features.npz"
    np.savez(npz_path, **features_dict)
    print(f"\nSaved PCA features to: {npz_path}")

    # Save PCA model
    model_path = output_dir / "lbp_pca_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"Saved PCA model to: {model_path}")

    # Save PCA info
    info_path = output_dir / "lbp_pca_info.txt"
    with open(info_path, "w") as f:
        f.write(f"Number of samples: {len(Xred)}\n")
        f.write(f"Original dimension: {original_dim}\n")
        f.write(f"Reduced dimension: {Xred.shape[1]}\n")
        f.write(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}\n")
        f.write("\nClasses:\n")
        for cls in np.unique(y):
            f.write(f"  {cls}: {np.sum(y == cls)}\n")

    print(f"Saved PCA info to: {info_path}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Apply PCA to LBP features")
    parser.add_argument(
        "--features-dir",
        type=str,
        default="output/lbp/features",
        help="Directory containing LBP features",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/lbp_pca",
        help="Directory to save PCA output",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=500,
        help="Maximum PCA components",
    )

    args = parser.parse_args()

    features_file = Path(args.features_dir) / "lbp_features.npz"
    if not features_file.exists():
        print(f"ERROR: LBP feature file not found at {features_file}")
        return

    # Load features
    X, y, filenames = load_lbp_features(features_file)
    original_dim = X.shape[1]

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Proper number of PCA components
    n_comp = min(args.n_components, X.shape[0], X.shape[1])
    print(f"\nApplying PCA with up to {n_comp} components...")

    pca = PCA(n_components=n_comp)
    Xred = pca.fit_transform(X_scaled)

    print(f"\nReduced dimensions: {Xred.shape[1]}")
    print(f"Explained variance sum: {pca.explained_variance_ratio_.sum():.4f}")

    # Save results
    save_pca_results(Xred, y, filenames, pca, args.output_dir, original_dim)

    print("\n" + "=" * 60)
    print("LBP PCA PROCESSING COMPLETE")
    print(f"Dimensionality reduced: {original_dim} → {Xred.shape[1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
