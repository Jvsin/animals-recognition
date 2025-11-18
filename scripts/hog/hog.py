"""
HOG (Histogram of Oriented Gradients) Feature Extraction Script
This script processes images from the dataset and extracts HOG features.
"""

import cv2
import numpy as np
from skimage.feature import hog
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# Import rescale_image from scaler.py
try:
    from scripts.scaler import rescale_image
except Exception as e:
    raise ImportError(
        f"Could not import 'scaler' module from sys.path: {e}"
    )


# =====================================================================
# HOG Feature Extraction Class
# =====================================================================
class HOGTransformer:
    """
    Class for extracting HOG features from images.
    """

    def __init__(self,
                 orientations=9,
                 pixels_per_cell=(16, 16),
                 cells_per_block=(2, 2),
                 visualize=True,
                 multichannel=False):

        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.visualize = visualize
        self.multichannel = multichannel

    def extract_hog_features(self, image):
        """Extract HOG features from an image."""

        # Skalowanie jak w innych metodach
        image = rescale_image(image)

        # uint8 dla bezpieczeństwa
        if image.dtype in (np.float32, np.float64):
            image = (image * 255).astype(np.uint8)

        # Konwersja do grayscale zgodnie z BGR → RGB → Gray
        if len(image.shape) == 3 and not self.multichannel:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if self.visualize:
            features, hog_image = hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                visualize=True,
                channel_axis=-1 if self.multichannel else None
            )
            return features, hog_image

        else:
            features = hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                visualize=False,
                channel_axis=-1 if self.multichannel else None
            )
            return features, None

    def process_image(self, image_path, output_dir=None, save_visualization=False):
        """Process single image."""

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error reading image: {image_path}")
            return None

        # BGR → RGB (brakowało!)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        features, hog_image = self.extract_hog_features(image)

        # Zapis wizualizacji
        if save_visualization and hog_image is not None and output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = Path(image_path).stem + "_hog.png"

            hog_norm = (hog_image * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / filename), hog_norm)

        return features

    def process_dataset(self,
                        dataset_dir,
                        photos_output_dir=None,
                        features_output_dir=None,
                        save_features=True,
                        save_visualizations=False,
                        limit=None,
                        split="train"):
        """Process entire dataset split."""

        dataset_path = Path(dataset_dir)
        split_path = dataset_path / split

        if not split_path.exists():
            print(f"Error: Split '{split}' not found in dataset.")
            return {}

        # Wczytywanie obrazów
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_files.extend(split_path.glob(ext))
            image_files.extend(split_path.glob(ext.upper()))

        print(f"Found {len(image_files)} images in '{split}' split.")

        if limit:
            image_files = image_files[:limit]
            print(f"Processing only {limit} images (limit).")

        all_features = {}

        for img_path in tqdm(
            image_files,
            desc="Extracting HOG features",
            unit="img",
            bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ):
            filename = img_path.stem
            class_name = "".join(c for c in filename if not c.isdigit()).rstrip("_")

            vis_dir = None
            if save_visualizations and photos_output_dir:
                vis_dir = Path(photos_output_dir) / class_name

            feats = self.process_image(
                img_path,
                output_dir=vis_dir,
                save_visualization=save_visualizations
            )

            if feats is not None:
                key = f"{class_name}/{img_path.name}"
                all_features[key] = feats

        # Zapis cech
        if save_features and features_output_dir and all_features:
            out_dir = Path(features_output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            file_npz = out_dir / "hog_features.npz"
            np.savez(file_npz, **all_features)
            print(f"\nSaved HOG features to {file_npz}")

            first_vector = next(iter(all_features.values()))
            info_file = out_dir / "hog_info.txt"
            with open(info_file, "w") as f:
                f.write(f"Number of images: {len(all_features)}\n")
                f.write(f"Feature vector length: {len(first_vector)}\n")
                f.write(f"Parameters:\n")
                f.write(f"  - Orientations: {self.orientations}\n")
                f.write(f"  - Pixels per cell: {self.pixels_per_cell}\n")
                f.write(f"  - Cells per block: {self.cells_per_block}\n")

            print(f"Saved feature info to {info_file}")

        return all_features


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Extract HOG features")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--photos-output", type=str, default="output/hog/photos")
    parser.add_argument("--features-output", type=str, default="output/hog/features")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--orientations", type=int, default=9)
    parser.add_argument("--pixels-per-cell", type=int, nargs=2, default=[8, 8])
    parser.add_argument("--cells-per-block", type=int, nargs=2, default=[2, 2])
    parser.add_argument("--split", type=str, choices=["train", "test", "valid"], default="train")

    args = parser.parse_args()

    transformer = HOGTransformer(
        orientations=args.orientations,
        pixels_per_cell=tuple(args.pixels_per_cell),
        cells_per_block=tuple(args.cells_per_block),
        visualize=args.visualize,   # SPÓJNE Z ORB/SIFT
        multichannel=False
    )

    features = transformer.process_dataset(
        dataset_dir=args.dataset_dir,
        photos_output_dir=args.photos_output,
        features_output_dir=args.features_output,
        save_features=True,
        save_visualizations=args.visualize,
        limit=args.limit,
        split=args.split
    )

    print(f"\nProcessing complete! Extracted features from {len(features)} images.")


if __name__ == "__main__":
    main()
