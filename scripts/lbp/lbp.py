"""
LBP (Local Binary Patterns) Feature Extraction Script
Extracts LBP histograms from images in the dataset.
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.feature import local_binary_pattern
import argparse
from tqdm import tqdm
import sys

# Load scaler
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scaler import rescale_image


# =====================================================================
# LBP Transformer
# =====================================================================
class LBPTransformer:
    """
    Extracts Local Binary Patterns features (histograms).
    """

    def __init__(self,
                 radius=3,
                 n_points=24,
                 method="uniform",
                 visualize=False):
        self.radius = radius
        self.n_points = n_points
        self.method = method
        self.visualize = visualize

        # Number of bins for uniform LBP
        self.hist_size = n_points + 2

    def extract_lbp_features(self, image):
        """Compute LBP histogram for a single image."""
        # skalowanie tak jak w innych metodach
        image = rescale_image(image)

        # KONWERSJA float → uint8 (ważne dla cv2.cvtColor)
        if image.dtype in (np.float32, np.float64):
            image = (image * 255).astype(np.uint8)

        # Convert to grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Compute LBP image
        lbp = local_binary_pattern(
            image,
            P=self.n_points,
            R=self.radius,
            method=self.method
        )

        # Histogram (znormalizowany)
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=self.hist_size,
            range=(0, self.hist_size),
            density=True
        )

        if self.visualize:
            return hist.astype(np.float32), lbp
        else:
            return hist.astype(np.float32), None

    def process_image(self, image_path, output_dir=None, save_visualization=False):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error reading {image_path}")
            return None

        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        features, lbp_img = self.extract_lbp_features(img)

        if save_visualization and lbp_img is not None and output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            name = image_path.stem + "_lbp.png"

            # normalizacja LBP do 0–255
            max_val = lbp_img.max()
            if max_val > 0:
                vis = (lbp_img / max_val * 255).astype(np.uint8)
            else:
                vis = np.zeros_like(lbp_img, dtype=np.uint8)

            cv2.imwrite(str(output_dir / name), vis)

        return features

    def process_dataset(self,
                        dataset_dir,
                        photos_output_dir=None,
                        features_output_dir=None,
                        save_features=True,
                        save_visualizations=False,
                        limit=None,
                        split="train"):

        dataset_path = Path(dataset_dir)
        split_path = dataset_path / split

        if not split_path.exists():
            print(f"Split '{split}' not found!")
            return {}

        files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            files.extend(split_path.glob(ext))
            files.extend(split_path.glob(ext.upper()))

        print(f"Found {len(files)} images in split '{split}'")

        if limit is not None:
            files = files[:limit]
            print(f"Processing only {limit} images (limit).")

        all_features = {}

        for img_path in tqdm(files, desc="Extracting LBP", unit="img"):
            filename = img_path.stem
            class_name = "".join(c for c in filename if not c.isdigit()).rstrip("_")

            vis_output = None
            if save_visualizations and photos_output_dir:
                vis_output = Path(photos_output_dir) / class_name

            feats = self.process_image(
                img_path,
                output_dir=vis_output,
                save_visualization=save_visualizations
            )

            if feats is not None:
                key = f"{class_name}/{img_path.name}"
                all_features[key] = feats

        if save_features and features_output_dir and all_features:
            out = Path(features_output_dir)
            out.mkdir(parents=True, exist_ok=True)

            fv = out / "lbp_features.npz"
            np.savez(fv, **all_features)
            print(f"\nSaved LBP features to {fv}")

            info = out / "lbp_info.txt"
            with open(info, "w") as f:
                f.write(f"Images: {len(all_features)}\n")
                f.write(f"Feature length: {len(next(iter(all_features.values())))}\n")
                f.write(f"Radius: {self.radius}\n")
                f.write(f"Points: {self.n_points}\n")
                f.write(f"Method: {self.method}\n")
            print(f"Saved info to {info}")

        return all_features


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Extract LBP features")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--photos-output", type=str, default="output/lbp/photos")
    parser.add_argument("--features-output", type=str, default="output/lbp/features")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--points", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", type=str, default="train")

    args = parser.parse_args()

    transformer = LBPTransformer(
        radius=args.radius,
        n_points=args.points,
        visualize=args.visualize
    )

    feats = transformer.process_dataset(
        dataset_dir=args.dataset_dir,
        photos_output_dir=args.photos_output,
        features_output_dir=args.features_output,
        save_features=True,
        save_visualizations=args.visualize,
        limit=args.limit,
        split=args.split
    )

    print(f"\nDONE. Extracted LBP features from {len(feats)} images.")


if __name__ == "__main__":
    main()
