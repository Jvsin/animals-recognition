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
    
    def process_dataset(self, dataset_dir, output_dir=None, save_features=True, 
                       save_visualizations=False, limit=None):
        """
        Process all images in a dataset directory.
        
        Args:
            dataset_dir: Directory containing images
            output_dir: Directory to save outputs
            save_features: Whether to save feature vectors
            save_visualizations: Whether to save HOG visualizations
            limit: Maximum number of images to process (for testing)
            
        Returns:
            all_features: Dictionary mapping image paths to feature vectors
        """
        dataset_path = Path(dataset_dir)
        all_features = {}
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_path.rglob(f'*{ext}'))
            image_files.extend(dataset_path.rglob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images in {dataset_dir}")
        
        if limit:
            image_files = image_files[:limit]
            print(f"Processing only {limit} images (limit applied)")
        
        # Process each image
        for idx, image_path in enumerate(image_files, 1):
            print(f"Processing {idx}/{len(image_files)}: {image_path.name}")
            
            features = self.process_image(
                image_path, 
                output_dir=output_dir if save_visualizations else None,
                save_visualization=save_visualizations
            )
            
            if features is not None:
                all_features[str(image_path)] = features
        
        # Save features to file if requested
        if save_features and output_dir and all_features:
            output_path = Path(output_dir)
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


def process_single_image(image_path, output_dir='output/hog_test', 
                         orientations=9, pixels_per_cell=(8, 8), 
                         cells_per_block=(2, 2)):
    """
    Process a single test image and display/save HOG visualization.
    
    Args:
        image_path: Path to the test image
        output_dir: Directory to save output
        orientations: Number of orientation bins
        pixels_per_cell: Size of a cell in pixels
        cells_per_block: Number of cells in each block
    """
    # Create HOG transformer
    transformer = HOGTransformer(
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
        multichannel=False
    )
    
    # Read and process image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Extract HOG features
    features, hog_image = transformer.extract_hog_features(image)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save only HOG image without any labels or comparisons
    filename = Path(image_path).stem + '_hog.png'
    output_file = output_path / filename
    
    # Convert HOG image to proper format and save directly
    hog_image_normalized = (hog_image * 255).astype(np.uint8)
    cv2.imwrite(str(output_file), hog_image_normalized)
    
    # Save features
    features_file = output_path / (Path(image_path).stem + '_hog_features.npy')
    np.save(features_file, features)
    
    print(f"Processed: {image_path}")


def main():
    """Main function to run HOG feature extraction."""
    parser = argparse.ArgumentParser(
        description='Extract HOG features from images in dataset'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'dataset'],
        default='dataset',
        help='Processing mode: single image or entire dataset'
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to single test image (required for single mode)'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset',
        help='Directory containing dataset images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/hog',
        help='Directory to save HOG features and visualizations'
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
    
    # Process single image or entire dataset
    if args.mode == 'single':
        if not args.image:
            parser.error("--image is required when using --mode single")
        
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            return
        
        # Process single image
        process_single_image(
            image_path=args.image,
            output_dir=args.output_dir,
            orientations=args.orientations,
            pixels_per_cell=tuple(args.pixels_per_cell),
            cells_per_block=tuple(args.cells_per_block)
        )
    
    else:  # dataset mode
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
            output_dir=args.output_dir,
            save_features=True,
            save_visualizations=args.visualize,
            limit=args.limit
        )
        
        print(f"Processing complete! Extracted features from {len(features)} images.")


if __name__ == '__main__':
    main()
