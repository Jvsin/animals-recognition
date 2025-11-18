"""
ORB (Oriented FAST and Rotated BRIEF) Feature Extraction Script
This script processes images from the dataset and extracts ORB features.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import random

# ============================================
# Import rescale_image z scaler.py (tak jak w HOG)
# ============================================

try:
    from scripts.scaler import rescale_image
except Exception as e:
    raise ImportError(
        f"Could not import 'scaler' module from sys.path: {e}"
    )


class ORBTransformer:
    """
    Class for extracting ORB features from images.
    """

    def __init__(self, n_keypoints=500, visualize=True):
        """
        Args:
            n_keypoints: docelowa liczba punktów kluczowych ORB
            visualize: czy zapisywać obraz z zaznaczonymi punktami
        """
        self.n_keypoints = n_keypoints
        self.visualize = visualize

        # Tworzenie detektora ORB w sposób kompatybilny z różnymi wersjami OpenCV
        orb_create = getattr(cv2, "ORB_create", None)
        if orb_create is not None:
            self.orb = orb_create(nfeatures=n_keypoints)
        else:
            orb_cls = getattr(cv2, "ORB", None)
            if orb_cls is not None and hasattr(orb_cls, "create"):
                self.orb = orb_cls.create(nfeatures=n_keypoints)
            else:
                raise AttributeError(
                    "cv2.ORB_create or cv2.ORB.create not found; "
                    "ensure 'opencv-python' is installed"
                )

    def extract_orb_features(self, image):
        """
        Extract ORB features from an image.

        Zwracamy wektor stałej długości:
        n_keypoints * 32 (bo ORB ma deskryptor 32-elementowy).
        Jeśli jest mniej punktów – dopadujemy zerami.
        Jeśli więcej – bierzemy pierwsze n_keypoints.
        """
        # Przeskaluj obraz tak jak w HOG
        image = rescale_image(image)
        image = np.asarray(image)

        # Na wszelki wypadek rzutowanie na uint8 jeśli typ jest float
        # (zakładamy wartości w zakresie [0,1] dla floatów)
        if np.issubdtype(image.dtype, np.floating):
            image = (image * 255).astype(np.uint8)

        # Konwersja do odcieni szarości
        if len(image.shape) == 3:
            # cv2 pracuje w BGR, więc formalnie używamy COLOR_BGR2GRAY
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        # Detekcja punktów kluczowych i obliczenie deskryptorów
        keypoints, descriptors = self.orb.detectAndCompute(image_gray, None)

        if descriptors is None:
            # Brak wykrytych punktów → deskryptory puste
            descriptors = np.zeros((0, 32), dtype=np.uint8)

        # Przycinanie / dopadanie do n_keypoints
        if descriptors.shape[0] > self.n_keypoints:
            descriptors = descriptors[:self.n_keypoints]
        elif descriptors.shape[0] < self.n_keypoints:
            pad = np.zeros(
                (self.n_keypoints - descriptors.shape[0], descriptors.shape[1]),
                dtype=descriptors.dtype,
            )
            descriptors = np.vstack([descriptors, pad])

        # Spłaszczenie do 1D wektora cech
        features = descriptors.flatten().astype(np.float32)

        vis_image = None
        if self.visualize:
            # Przygotowanie obrazu wyjściowego (BGR), drawKeypoints oczekuje 3 kanałów
            if len(image_gray.shape) == 2:
                out_img = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
            else:
                out_img = image_gray.copy()

            vis_image = cv2.drawKeypoints(
                out_img,
                keypoints,
                out_img,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

        return features, vis_image

    def process_image(self, image_path, output_dir=None, save_visualization=False):
        """
        Process a single image and extract ORB features.
        """
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None

        features, vis_image = self.extract_orb_features(image)

        # Zapis wizualizacji
        if save_visualization and vis_image is not None and output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            filename = Path(image_path).stem + "_orb.png"
            cv2.imwrite(str(output_path / filename), vis_image)

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
        """
        Process all images in a dataset directory structure.

        Zapisujemy features w pliku:
        output/orb/features/orb_features.npz
        (klucz: "class/filename" – tak samo jak w HOG).
        """
        dataset_path = Path(dataset_dir)
        all_features = {}

        split_path = dataset_path / split
        if not split_path.exists():
            print(f"Error: Split folder '{split}' not found in {dataset_dir}")
            return all_features

        # Zbieramy listę wszystkich obrazów
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(split_path.glob(f"*{ext}"))
            image_files.extend(split_path.glob(f"*{ext.upper()}"))

        print(f"Found {len(image_files)} images in '{split}' split")

        # Opcjonalne losowe ograniczenie liczby obrazów
        if limit:
            random.shuffle(image_files)
            image_files = image_files[:limit]
            print(f"Processing only {limit} randomly selected images (limit applied)")

        # Główna pętla po obrazach
        for image_path in tqdm(
            image_files,
            desc="Extracting ORB features",
            unit="img",
            bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            filename = image_path.stem
            # Wyciągamy nazwę klasy z prefiksu nazwy pliku (bez cyfr i podkreśleń na końcu)
            class_name = "".join(c for c in filename if not c.isdigit()).rstrip("_")

            vis_output_dir = None
            if save_visualizations and photos_output_dir:
                vis_output_dir = Path(photos_output_dir) / class_name

            features = self.process_image(
                image_path,
                output_dir=vis_output_dir,
                save_visualization=save_visualizations,
            )

            if features is not None:
                key = f"{class_name}/{image_path.name}"
                all_features[key] = features

        # Zapis wektorów cech do pliku NPZ + info do TXT
        if save_features and features_output_dir and all_features:
            output_path = Path(features_output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            features_file = output_path / "orb_features.npz"
            np.savez(features_file, **all_features)
            print(f"\nSaved ORB features to {features_file}")

            first_feature = next(iter(all_features.values()))
            info_file = output_path / "orb_info.txt"
            with open(info_file, "w") as f:
                f.write(f"Number of images: {len(all_features)}\n")
                f.write(f"Feature vector length: {len(first_feature)}\n")
                f.write("ORB parameters:\n")
                f.write(f"  - n_keypoints: {self.n_keypoints}\n")
                f.write("Descriptor length per keypoint: 32\n")
            print(f"Saved feature info to {info_file}")

        return all_features


def main():
    """Main function to run ORB feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract ORB features from images in dataset"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Directory containing dataset with class folders",
    )
    parser.add_argument(
        "--photos-output",
        type=str,
        default="output/orb/photos",
        help="Directory to save ORB visualization images",
    )
    parser.add_argument(
        "--features-output",
        type=str,
        default="output/orb/features",
        help="Directory to save ORB feature vectors",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save ORB keypoint visualizations for each image",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (for testing)",
    )
    parser.add_argument(
        "--n-keypoints",
        type=int,
        default=500,
        help="Number of ORB keypoints per image",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "valid"],
        help="Which dataset split to process (train, test, or valid)",
    )

    args = parser.parse_args()

    transformer = ORBTransformer(
        n_keypoints=args.n_keypoints,
        visualize=args.visualize,
    )

    features = transformer.process_dataset(
        dataset_dir=args.dataset_dir,
        photos_output_dir=args.photos_output,
        features_output_dir=args.features_output,
        save_features=True,
        save_visualizations=args.visualize,
        limit=args.limit,
        split=args.split,
    )

    print(f"\nProcessing complete! Extracted ORB features from {len(features)} images.")


if __name__ == "__main__":
    main()
