"""
SIFT (Scale-Invariant Feature Transform) Feature Extraction Script
This script processes images from the dataset and extracts SIFT keypoints and descriptors.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse


class SIFTTransformer:
    """
    Class for extracting SIFT features (keypoints & descriptors) from images.
    """

    def __init__(
        self,
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
        visualize=True,
        draw_rich=True,
    ):
        """
        Initialize SIFT transformer with parameters.

        Args:
            nfeatures: The number of best features to retain (0 = all)
            nOctaveLayers: Number of layers in each octave
            contrastThreshold: Contrast threshold
            edgeThreshold: Edge threshold
            sigma: Sigma of the Gaussian applied to the input image at the octave #0
            visualize: Whether to save visualization images with drawn keypoints
            draw_rich: Use rich keypoint drawing (size & orientation)
        """
        # Make sure SIFT is available (opencv-contrib-python)
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError(
                "SIFT is not available in your OpenCV build. "
                "Install opencv-contrib-python: pip install opencv-contrib-python"
            )

        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma,
        )
        self.visualize = visualize
        self.draw_rich = draw_rich

        # Keep params for logging/saving
        self.params = dict(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma,
            visualize=visualize,
            draw_rich=draw_rich,
        )

    def extract_sift(self, image):
        """
        Extract SIFT keypoints and descriptors from an image.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            keypoints: list of cv2.KeyPoint
            descriptors: np.ndarray of shape (N, 128) or empty array with (0, 128)
        """
        # Convert to grayscale
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        keypoints, descriptors = self.sift.detectAndCompute(gray, None)

        if descriptors is None:
            descriptors = np.empty((0, 128), dtype=np.float32)

        return keypoints, descriptors

    def _draw_keypoints(self, image, keypoints):
        if image.ndim == 2:
            img_for_draw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img_for_draw = image.copy()

        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if self.draw_rich else 0
        vis = cv2.drawKeypoints(img_for_draw, keypoints, None, flags=flags)
        return vis

    def process_image(self, image_path, output_dir=None, save_visualization=False):
        """
        Process a single image and extract SIFT descriptors.

        Args:
            image_path: Path to the input image
            output_dir: Directory to save visualization (optional)
            save_visualization: Whether to save visualization image

        Returns:
            descriptors: np.ndarray (N, 128)
            num_keypoints: int
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None, 0

        keypoints, descriptors = self.extract_sift(image)

        if save_visualization and output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            vis_img = self._draw_keypoints(image, keypoints)
            filename = Path(image_path).stem + "_sift.png"
            cv2.imwrite(str(Path(output_dir) / filename), vis_img)

        return descriptors, len(keypoints)

    def process_dataset(
        self,
        dataset_dir,
        photos_output_dir=None,
        features_output_dir=None,
        save_features=True,
        save_visualizations=False,
        limit=None,
    ):
        """
        Process all images in a dataset directory structure.

        Args:
            dataset_dir: Directory containing dataset with class folders
            photos_output_dir: Directory to save SIFT visualization images
            features_output_dir: Directory to save descriptors
            save_features: Whether to save descriptors file
            save_visualizations: Whether to save keypoint visualizations
            limit: Maximum number of images to process (for testing)

        Returns:
            results: dict mapping "class/filename" -> dict with:
                     {"count": int, "desc": np.ndarray (N, 128)}
        """
        dataset_path = Path(dataset_dir)
        results = {}

        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        print(f"Found {len(class_dirs)} class directories")

        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []

        for class_dir in class_dirs:
            images_path = class_dir / "train" / "images"
            if images_path.exists():
                for ext in image_extensions:
                    image_files.extend(images_path.glob(f"*{ext}"))
                    image_files.extend(images_path.glob(f"*{ext.upper()}"))

        print(f"Found {len(image_files)} images total")

        if limit:
            image_files = image_files[:limit]
            print(f"Processing only {limit} images (limit applied)")

        # Process
        for idx, image_path in enumerate(image_files, 1):
            class_name = image_path.parent.parent.parent.name
            print(f"Processing {idx}/{len(image_files)}: {class_name}/{image_path.name}")

            vis_output_dir = None
            if save_visualizations and photos_output_dir:
                vis_output_dir = Path(photos_output_dir) / class_name

            desc, count = self.process_image(
                image_path, output_dir=vis_output_dir, save_visualization=save_visualizations
            )

            if desc is not None:
                key = f"{class_name}/{image_path.name}"
                results[key] = {"count": int(count), "desc": desc.astype(np.float32)}

        # Save features
        if save_features and features_output_dir and results:
            out_dir = Path(features_output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            keys = list(results.keys())
            counts = np.array([results[k]["count"] for k in keys], dtype=np.int32)
            desc_obj = np.empty(len(keys), dtype=object)
            for i, k in enumerate(keys):
                desc_obj[i] = results[k]["desc"]

            features_file = out_dir / "sift_features.npz"
            np.savez_compressed(
                features_file,
                names=np.array(keys),
                counts=counts,
                desc=desc_obj,
                **{f"param_{k}": v for k, v in self.params.items()},
            )
            print(f"\nSaved SIFT features to {features_file}")

            info_file = out_dir / "sift_info.txt"
            with open(info_file, "w") as f:
                f.write(f"Number of images: {len(keys)}\n")
                total_kp = int(counts.sum())
                f.write(f"Total keypoints: {total_kp}\n")
                f.write(f"Average keypoints per image: {total_kp/len(keys):.2f}\n")
                f.write("SIFT parameters:\n")
                for k, v in self.params.items():
                    f.write(f"  - {k}: {v}\n")
            print(f"Saved feature info to {info_file}")

        return results


def main():
    """Main function to run SIFT feature extraction."""
    parser = argparse.ArgumentParser(description="Extract SIFT features from images in dataset")
    parser.add_argument(
        "--dataset-dir", type=str, default="dataset", help="Directory containing dataset with class folders"
    )
    parser.add_argument(
        "--photos-output",
        type=str,
        default="output/sift/photos",
        help="Directory to save SIFT keypoint visualization images",
    )
    parser.add_argument(
        "--features-output",
        type=str,
        default="output/sift/features",
        help="Directory to save SIFT descriptors",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Save SIFT keypoint visualizations for each image"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process (for testing)")

    # SIFT params
    parser.add_argument("--nfeatures", type=int, default=0, help="Number of best features to retain (0=all)")
    parser.add_argument("--nOctaveLayers", type=int, default=3, help="Number of layers in each octave")
    parser.add_argument("--contrastThreshold", type=float, default=0.04, help="Contrast threshold")
    parser.add_argument("--edgeThreshold", type=float, default=10, help="Edge threshold")
    parser.add_argument("--sigma", type=float, default=1.6, help="Sigma of input image blur at octave #0")
    parser.add_argument("--no-rich", action="store_true", help="Disable rich keypoint drawing")

    args = parser.parse_args()

    transformer = SIFTTransformer(
        nfeatures=args.nfeatures,
        nOctaveLayers=args.nOctaveLayers,
        contrastThreshold=args.contrastThreshold,
        edgeThreshold=args.edgeThreshold,
        sigma=args.sigma,
        visualize=True,
        draw_rich=not args.no_rich,
    )

    results = transformer.process_dataset(
        dataset_dir=args.dataset_dir,
        photos_output_dir=args.photos_output,
        features_output_dir=args.features_output,
        save_features=True,
        save_visualizations=args.visualize,
        limit=args.limit,
    )

    print(f"\nProcessing complete! Extracted SIFT from {len(results)} images.")


if __name__ == "__main__":
    main()
