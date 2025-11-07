"""
HOG (Histogram of Oriented Gradients) Feature Extraction Script
This script processes images from the dataset and extracts HOG features.
"""

import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


class HOGTransformer:
    """
    Class for extracting HOG features from images.
    """
    
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), 
                 cells_per_block=(2, 2), visualize=True, multichannel=False):
        """
        Initialize HOG transformer with parameters.
        
        Args:
            orientations: Number of orientation bins
            pixels_per_cell: Size of a cell in pixels
            cells_per_block: Number of cells in each block
            visualize: Whether to return visualization image
            multichannel: Whether to process color images
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.visualize = visualize
        self.multichannel = multichannel
    
    def extract_hog_features(self, image):
        """
        Extract HOG features from an image.
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            features: HOG feature vector
            hog_image: HOG visualization image (if visualize=True)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3 and not self.multichannel:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract HOG features
        if self.visualize:
            features, hog_image = hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                visualize=self.visualize,
                channel_axis=-1 if self.multichannel else None
            )
            
            # Rescale histogram for better display
            hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            
            return features, hog_image
        else:
            features = hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                visualize=self.visualize,
                channel_axis=-1 if self.multichannel else None
            )
            return features, None
    
    def process_image(self, image_path, output_dir=None, save_visualization=False):
        """
        Process a single image and extract HOG features.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save HOG visualization (optional)
            save_visualization: Whether to save visualization image
            
        Returns:
            features: HOG feature vector
        """
        # Read image
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None
        
        # Extract HOG features
        features, hog_image = self.extract_hog_features(image)
        
        # Save visualization if requested
        if save_visualization and hog_image is not None and output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save only HOG image without any labels or axes
            filename = Path(image_path).stem + '_hog.png'
            
            # Convert HOG image to proper format and save directly
            hog_image_normalized = (hog_image * 255).astype(np.uint8)
            cv2.imwrite(str(output_path / filename), hog_image_normalized)
        
        return features
    
    def process_dataset(self, dataset_dir, photos_output_dir=None, features_output_dir=None, 
                       save_features=True, save_visualizations=False, limit=None):
        """
        Process all images in a dataset directory structure.
        
        Args:
            dataset_dir: Directory containing dataset with class folders
            photos_output_dir: Directory to save HOG visualization images
            features_output_dir: Directory to save feature vectors
            save_features: Whether to save feature vectors
            save_visualizations: Whether to save HOG visualizations
            limit: Maximum number of images to process (for testing)
            
        Returns:
            all_features: Dictionary mapping image paths to feature vectors
        """
        dataset_path = Path(dataset_dir)
        all_features = {}
        
        # Find all class directories
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        print(f"Found {len(class_dirs)} class directories")
        
        # Find all image files in train/images subdirectories
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for class_dir in class_dirs:
            images_path = class_dir / 'train' / 'images'
            if images_path.exists():
                for ext in image_extensions:
                    image_files.extend(images_path.glob(f'*{ext}'))
                    image_files.extend(images_path.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images total")
        
        if limit:
            image_files = image_files[:limit]
            print(f"Processing only {limit} images (limit applied)")
        
        # Process each image
        for idx, image_path in enumerate(image_files, 1):
            # Get class name from directory structure
            class_name = image_path.parent.parent.parent.name
            
            print(f"Processing {idx}/{len(image_files)}: {class_name}/{image_path.name}")
            
            # Set output directory for visualizations (by class)
            vis_output_dir = None
            if save_visualizations and photos_output_dir:
                vis_output_dir = Path(photos_output_dir) / class_name
            
            features = self.process_image(
                image_path, 
                output_dir=vis_output_dir,
                save_visualization=save_visualizations
            )
            
            if features is not None:
                # Store with class name and image name
                key = f"{class_name}/{image_path.name}"
                all_features[key] = features
        
        # Save features to file if requested
        if save_features and features_output_dir and all_features:
            output_path = Path(features_output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save as numpy file
            features_file = output_path / 'hog_features.npz'
            np.savez(features_file, **all_features)
            print(f"\nSaved HOG features to {features_file}")
            
            # Save feature shape info
            first_feature = next(iter(all_features.values()))
            info_file = output_path / 'hog_info.txt'
            with open(info_file, 'w') as f:
                f.write(f"Number of images: {len(all_features)}\n")
                f.write(f"Feature vector length: {len(first_feature)}\n")
                f.write(f"HOG parameters:\n")
                f.write(f"  - Orientations: {self.orientations}\n")
                f.write(f"  - Pixels per cell: {self.pixels_per_cell}\n")
                f.write(f"  - Cells per block: {self.cells_per_block}\n")
            print(f"Saved feature info to {info_file}")
        
        return all_features


def main():
    """Main function to run HOG feature extraction."""
    parser = argparse.ArgumentParser(
        description='Extract HOG features from images in dataset'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset',
        help='Directory containing dataset with class folders'
    )
    parser.add_argument(
        '--photos-output',
        type=str,
        default='output/hog/photos',
        help='Directory to save HOG visualization images'
    )
    parser.add_argument(
        '--features-output',
        type=str,
        default='output/hog/features',
        help='Directory to save feature vectors'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Save HOG visualizations for each image'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of images to process (for testing)'
    )
    parser.add_argument(
        '--orientations',
        type=int,
        default=9,
        help='Number of orientation bins for HOG'
    )
    parser.add_argument(
        '--pixels-per-cell',
        type=int,
        nargs=2,
        default=[8, 8],
        help='Size of a cell in pixels (height width)'
    )
    parser.add_argument(
        '--cells-per-block',
        type=int,
        nargs=2,
        default=[2, 2],
        help='Number of cells in each block (height width)'
    )
    
    args = parser.parse_args()
    
    # Create HOG transformer
    transformer = HOGTransformer(
        orientations=args.orientations,
        pixels_per_cell=tuple(args.pixels_per_cell),
        cells_per_block=tuple(args.cells_per_block),
        visualize=True,
        multichannel=False
    )
    
    # Process dataset
    features = transformer.process_dataset(
        dataset_dir=args.dataset_dir,
        photos_output_dir=args.photos_output,
        features_output_dir=args.features_output,
        save_features=True,
        save_visualizations=args.visualize,
        limit=args.limit
    )
    
    print(f"\nProcessing complete! Extracted features from {len(features)} images.")


if __name__ == '__main__':
    main()
