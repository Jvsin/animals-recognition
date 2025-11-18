"""
Display SIFT feature statistics from NPZ file
"""

import numpy as np
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-file", type=str, default="output/sift/features/sift_features.npz")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--show-vector", action="store_true")
    args = parser.parse_args()

    data = np.load(args.features_file)

    print(f"\nLoaded {len(data.files)} entries")
    print("=======================================")

    for i, key in enumerate(list(data.files)[:args.limit]):
        vec = data[key]
        print(f"[{i+1}] {key}")
        print(f"  Shape: {vec.shape}")
        print(f"  Min: {vec.min():.4f}, Max: {vec.max():.4f}, Mean: {vec.mean():.4f}")

        if args.show_vector:
            print("  First 10 values:", vec[:10])

        print()

    all_feats = np.array([data[k] for k in data.files])
    print("=======================================")
    print(f"Global shape: {all_feats.shape}")
    print(f"Global mean: {all_feats.mean():.4f}")
    print(f"Global std: {all_feats.std():.4f}")


if __name__ == "__main__":
    main()
