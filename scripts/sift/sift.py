"""
SIFT Feature Extraction Script
This script processes images from the dataset and extracts SIFT features.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import importlib.util

# Load scaler.py (tak jak w HOG/ORB)
scaler_path = Path(__file__).resolve().parent.parent / "scaler.py"
if scaler_path.is_file():
    spec = importlib.util.spec_from_file_location("scaler", str(scaler_path))
    scaler_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scaler_mod)
    rescale_image = scaler_mod.rescale_image
else:
    try:
        from scripts.scaler import rescale_image
    except Exception as e:
        raise ImportError(
            f"Could not import 'scaler' module from {scaler_path} or sys.path: {e}"
        )


class SIFTTransformer:
    """
    Extract SIFT features from images. Produces fixed-length descriptors.
    """

    def __init__(self, n_keypoints=500, visualize=True):
        self.n_keypoints = n_keypoints
        self.visualize = visualize

        sift_create = getattr(cv2, "SIFT_create", None)
        if sift_create is None:
            raise RuntimeError(
                "SIFT not available. Install opencv-contrib-python."
            )

        self.sift = sift_create(nfeatures=n_keypoints)

    def extract_sift_features(self, image):
        image = rescale_image(image)

        # Upewniamy się, że obraz jest w formacie uint8 (0–255),
        # bo rescale_image może zwrócić float64 w [0, 1]
        if image.dtype in (np.float32, np.float64):
            image = (image * 255).astype(np.uint8)

        # grayscale – cv2.imread zwraca BGR, ale po skalowaniu i tak jest OK,
        # ważne żeby typ był uint8
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        keypoints, descriptors = self.sift.detectAndCompute(gray, None)

        if descriptors is None:
            descriptors = np.zeros((0, 128), dtype=np.float32)

        # dopaduj do stałej liczby
        if descriptors.shape[0] > self.n_keypoints:
            descriptors = descriptors[:self.n_keypoints]
        elif descriptors.shape[0] < self.n_keypoints:
            pad = np.zeros((self.n_keypoints - descriptors.shape[0], 128), dtype=np.float32)
            descriptors = np.vstack([descriptors, pad])

        features = descriptors.flatten().astype(np.float32)

        # visual
        vis_img = None
        if self.visualize:
            out_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            vis_img = cv2.drawKeypoints(out_img, keypoints, out_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return features, vis_img

    def process_image(self, image_path, output_dir=None, save_visualization=False):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error reading {image_path}")
            return None

        features, vis = self.extract_sift_features(img)

        if save_visualization and vis is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            name = image_path.stem + "_sift.png"
            cv2.imwrite(str(output_dir / name), vis)

        return features

    def process_dataset(
        self,
        dataset_dir,
        photos_output_dir=None,
        features_output_dir=None,
        save_features=True,
        save_visualizations=False,
        limit=None,
        split="train",
    ):

        dataset_path = Path(dataset_dir)
        split_path = dataset_path / split

        if not split_path.exists():
            print(f"Split '{split}' not found in dataset.")
            return {}

        files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            files.extend(split_path.glob(ext))

        if limit is not None:
            files = files[:limit]

        all_features = {}

        for img_path in tqdm(files, desc="Extracting SIFT", unit="img"):
            filename = img_path.stem
            class_name = "".join([c for c in filename if not c.isdigit()]).rstrip("_")

            vis_output = None
            if save_visualizations:
                vis_output = Path(photos_output_dir) / class_name

            feats = self.process_image(
                img_path,
                output_dir=vis_output,
                save_visualization=save_visualizations,
            )

            if feats is not None:
                key = f"{class_name}/{img_path.name}"
                all_features[key] = feats

        if save_features and features_output_dir and all_features:
            out = Path(features_output_dir)
            out.mkdir(parents=True, exist_ok=True)
            file_npz = out / "sift_features.npz"
            np.savez(file_npz, **all_features)

            print(f"\nSaved SIFT features to {file_npz}")
            print(f"Vector length: {len(next(iter(all_features.values())))}")

        return all_features


def main():
    parser = argparse.ArgumentParser(description="Extract SIFT features")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--photos-output", type=str, default="output/sift/photos")
    parser.add_argument("--features-output", type=str, default="output/sift/features")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-keypoints", type=int, default=500)
    parser.add_argument("--split", type=str, default="train")

    args = parser.parse_args()

    transformer = SIFTTransformer(
        n_keypoints=args.n_keypoints,
        visualize=args.visualize
    )

    feats = transformer.process_dataset(
        dataset_dir=args.dataset_dir,
        photos_output_dir=args.photos_output,
        features_output_dir=args.features_output,
        save_features=True,
        save_visualizations=args.visualize,
        limit=args.limit,
        split=args.split,
    )

    print(f"\nDONE. Extracted SIFT features from {len(feats)} images.")


if __name__ == "__main__":
    main()
