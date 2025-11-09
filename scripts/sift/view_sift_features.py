"""
Simple script to view SIFT features saved in NPZ file
"""

import numpy as np
from pathlib import Path
import argparse


def view_features(features_file, limit=10, show_first_desc=False):
    """
    View SIFT features from NPZ file.

    The NPZ is expected to contain:
      - names: array of strings "class/filename"
      - counts: array of ints (keypoints per image)
      - desc: object array of shape (num_images,), where each element is np.ndarray (Ni, 128)

    Args:
        features_file: Path to NPZ file with SIFT descriptors
        limit: Number of entries to display
        show_first_desc: Whether to print first descriptor vector (first 10 values)
    """
    print(f"Loading features from: {features_file}")
    data = np.load(features_file, allow_pickle=True)

    names = data["names"]
    counts = data["counts"]
    desc_obj = data["desc"]  # dtype=object

    print(f"\n{'='*60}")
    print(f"SIFT Features Summary")
    print(f"{'='*60}")
    print(f"Total images: {len(names)}")
    print(f"Total keypoints: {int(counts.sum())}")
    print(f"Average keypoints per image: {counts.mean():.2f}")

    # Count images per class
    class_counts = {}
    for key in names:
        class_name = str(key).split("/")[0]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print(f"\nImages per class:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")

    # Show sample entries
    print(f"\n{'='*60}")
    print(f"Sample entries (showing first {min(limit, len(names))}):")
    print(f"{'='*60}")

    for idx in range(min(limit, len(names))):
        key = names[idx]
        desc = desc_obj[idx]
        shape = tuple(desc.shape) if isinstance(desc, np.ndarray) else None
        print(f"\n[{idx+1}] {key}")
        print(f"    Keypoints: {counts[idx]}")
        print(f"    Descriptors shape: {shape}")
        if show_first_desc and desc.size > 0:
            vec = desc[0]
            preview = np.array2string(vec[:10], precision=4, separator=", ")
            print(f"    First descriptor (10 vals): {preview}")

    # Global descriptor stats (concatenate safely if not too large)
    print(f"\n{'='*60}")
    print(f"Global Descriptor Statistics:")
    print(f"{'='*60}")

    # To avoid massive memory usage, compute stats incrementally
    total = 0
    mean_acc = np.zeros(128, dtype=np.float64)
    m2_acc = np.zeros(128, dtype=np.float64)  # for variance (Welford)

    for desc in desc_obj:
        if not isinstance(desc, np.ndarray) or desc.size == 0:
            continue
        # desc is (Ni, 128)
        for row in desc:
            total += 1
            delta = row - mean_acc
            mean_acc += delta / total
            delta2 = row - mean_acc
            m2_acc += delta2 * delta

    if total > 1:
        var = m2_acc / (total - 1)
        print(f"Total descriptors: {total}")
        print(f"Per-dimension mean (first 10): {np.array2string(mean_acc[:10], precision=4, separator=', ')}")
        print(f"Per-dimension std  (first 10): {np.array2string(np.sqrt(var[:10]), precision=4, separator=', ')}")
    else:
        print("Not enough descriptors to compute global statistics.")


def main():
    parser = argparse.ArgumentParser(description="View SIFT features from NPZ file")
    parser.add_argument(
        "--features-file",
        type=str,
        default="output/sift/features/sift_features.npz",
        help="Path to NPZ file with SIFT descriptors",
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of sample entries to display")
    parser.add_argument(
        "--show-first-desc",
        action="store_true",
        help="Show first descriptor vector (first 10 values) for each sample",
    )

    args = parser.parse_args()

    features_path = Path(args.features_file)
    if not features_path.exists():
        print(f"Error: Features file not found: {args.features_file}")
        return

    view_features(args.features_file, args.limit, args.show_first_desc)


if __name__ == "__main__":
    main()
