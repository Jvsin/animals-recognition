"""
Simple viewer for LBP .npz features
"""

import numpy as np
from pathlib import Path
import argparse


def view_features(path, limit=10, show_vector=False):

    print(f"Loading LBP features from: {path}")
    data = np.load(path)

    print("\n==================================")
    print("LBP Feature Summary")
    print("==================================")

    print(f"Total images: {len(data.files)}")

    first = data[data.files[0]]
    print(f"Feature length: {len(first)}")

    print("\nImages per class:")
    counts = {}
    for k in data.files:
        cls = k.split("/")[0]
        counts[cls] = counts.get(cls, 0) + 1
    for c, n in counts.items():
        print(f"  {c}: {n}")

    print("\nSample entries:")
    for i, k in enumerate(data.files[:limit]):
        f = data[k]
        print(f"\n[{i+1}] {k}")
        print(f"  min={f.min():.4f}, max={f.max():.4f}, mean={f.mean():.4f}")
        if show_vector:
            print("  first 10:", f[:10])

    print("\nGlobal stats:")
    all_vec = np.array([data[k] for k in data.files])
    print(f"Shape: {all_vec.shape}")
    print(f"Mean: {all_vec.mean():.4f}, Std: {all_vec.std():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-file", type=str,
                        default="output/lbp/features/lbp_features.npz")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--show-vector", action="store_true")
    args = parser.parse_args()

    path = Path(args.features_file)
    if not path.exists():
        print("File not found:", path)
        return

    view_features(path, args.limit, args.show_vector)


if __name__ == "__main__":
    main()
