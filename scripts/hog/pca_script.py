"""
PCA Script for HOG Features
Loads HOG features and applies PCA dimensionality reduction using pca.py
"""

import numpy as np
from pathlib import Path
import argparse
import pickle
import sys

# Import PCA function from pca.py
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pca import perform_pca, N_COMPONENTS


def load_hog_features(features_path):
    """
    Load HOG features from NPZ file.
    
    Args:
        features_path: Path to NPZ file with HOG features
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels array (n_samples,)
        filenames: List of image filenames
    """
    print(f"Loading HOG features from {features_path}...")
    
    data = np.load(features_path)
    
    X = []
    y = []
    filenames = []
    
    for key in data.files:
        # Key format: "class_name/image_filename.jpg"
        class_name = key.split('/')[0]
        filename = key.split('/')[1]
        
        features = data[key]
        
        X.append(features)
        y.append(class_name)
        filenames.append(filename)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features each")
    print(f"Classes: {np.unique(y)}")
    print(f"Samples per class:")
    for label in np.unique(y):
        count = np.sum(y == label)
        print(f"  {label}: {count}")
    
    return X, y, filenames


def save_pca_results(X_reduced, y, filenames, pca_model, output_dir):
    """
    Save PCA-reduced features and model.
    
    Args:
        X_reduced: Reduced feature matrix
        y: Labels array
        filenames: List of image filenames
        pca_model: Fitted PCA model
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save reduced features as NPZ
    features_dict = {}
    for i, (features, label, filename) in enumerate(zip(X_reduced, y, filenames)):
        key = f"{label}/{filename}"
        features_dict[key] = features
    
    features_file = output_path / 'pca_features.npz'
    np.savez(features_file, **features_dict)
    print(f"\nSaved PCA features to {features_file}")
    
    # Save PCA model
    pca_file = output_path / 'pca_model.pkl'
    with open(pca_file, 'wb') as f:
        pickle.dump(pca_model, f)
    print(f"Saved PCA model to {pca_file}")
    
    # Save info file
    info_file = output_path / 'pca_info.txt'
    with open(info_file, 'w') as f:
        f.write(f"Number of samples: {len(X_reduced)}\n")
        f.write(f"Reduced feature dimensions: {X_reduced.shape[1]}\n")
        f.write(f"PCA components: {N_COMPONENTS}\n")
        f.write(f"Explained variance ratio: {pca_model.explained_variance_ratio_.sum():.4f}\n")
        f.write(f"\nClasses: {list(np.unique(y))}\n")
        f.write(f"\nSamples per class:\n")
        for label in np.unique(y):
            count = np.sum(y == label)
            f.write(f"  {label}: {count}\n")
    print(f"Saved info to {info_file}")


def main():
    parser = argparse.ArgumentParser(description='Apply PCA to HOG features')
    parser.add_argument(
        '--features-dir',
        type=str,
        default='output/hog/features',
        help='Directory containing HOG features (hog_features.npz)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/pca',
        help='Directory to save PCA results'
    )
    
    args = parser.parse_args()
    
    # Load HOG features
    features_file = Path(args.features_dir) / 'hog_features.npz'
    if not features_file.exists():
        print(f"Error: Features file not found: {features_file}")
        return
    
    X, y, filenames = load_hog_features(features_file)
    
    print(f"\nOriginal feature dimensions: {X.shape[1]}")
    
    # Apply PCA using function from pca.py
    print(f"Applying PCA with {N_COMPONENTS} components...")
    X_reduced, pca_model = perform_pca(X)
    
    print(f"Reduced feature dimensions: {X_reduced.shape[1]}")
    print(f"Explained variance: {pca_model.explained_variance_ratio_.sum():.4f}")
    
    # Save results
    save_pca_results(X_reduced, y, filenames, pca_model, args.output_dir)
    
    print("\n" + "="*50)
    print("PCA processing complete!")
    print(f"Dimensionality reduced: {X.shape[1]} → {X_reduced.shape[1]}")
    print("="*50)


if __name__ == '__main__':
    main()
