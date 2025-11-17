"""
Simple script to view HOG features saved in NPZ file
"""

import numpy as np
from pathlib import Path
import argparse


def view_features(features_file, limit=10, show_vector=False):
    """
    View HOG features from NPZ file.
    
    Args:
        features_file: Path to NPZ file with features
        limit: Number of entries to display
        show_vector: Whether to show full feature vector
    """
    # Load features
    print(f"Loading features from: {features_file}")
    data = np.load(features_file)
    
    print(f"\n{'='*60}")
    print(f"HOG Features Summary")
    print(f"{'='*60}")
    print(f"Total images: {len(data.files)}")
    
    # Get first feature to check dimensions
    first_key = data.files[0]
    first_feature = data[first_key]
    print(f"Feature vector length: {len(first_feature)}")
    print(f"Feature vector shape: {first_feature.shape}")
    
    # Count images per class
    class_counts = {}
    for key in data.files:
        class_name = key.split('/')[0]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\nImages per class:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")
    
    # Show sample entries
    print(f"\n{'='*60}")
    print(f"Sample entries (showing first {limit}):")
    print(f"{'='*60}")
    
    for idx, key in enumerate(data.files[:limit]):
        features = data[key]
        print(f"\n[{idx+1}] {key}")
        print(f"    Shape: {features.shape}")
        print(f"    Min: {features.min():.4f}, Max: {features.max():.4f}, Mean: {features.mean():.4f}")
        
        if show_vector:
            print(f"    First 10 values: {features[:10]}")
    
    # Statistics
    print(f"\n{'='*60}")
    print(f"Global Statistics:")
    print(f"{'='*60}")
    
    all_vectors = [data[key] for key in data.files]
    all_data = np.array(all_vectors)
    
    print(f"All features shape: {all_data.shape}")
    print(f"Overall min: {all_data.min():.4f}")
    print(f"Overall max: {all_data.max():.4f}")
    print(f"Overall mean: {all_data.mean():.4f}")
    print(f"Overall std: {all_data.std():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='View HOG features from NPZ file'
    )
    parser.add_argument(
        '--features-file',
        type=str,
        default='output/hog/features/hog_features.npz',
        help='Path to NPZ file with HOG features'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Number of sample entries to display'
    )
    parser.add_argument(
        '--show-vector',
        action='store_true',
        help='Show first 10 values of each feature vector'
    )
    
    args = parser.parse_args()
    
    features_path = Path(args.features_file)
    
    if not features_path.exists():
        print(f"Error: Features file not found: {args.features_file}")
        return
    
    view_features(args.features_file, args.limit, args.show_vector)


if __name__ == '__main__':
    main()
