"""
Script to analyze image sizes in the dataset
"""

import cv2
from pathlib import Path
from collections import Counter
import numpy as np

def analyze_image_sizes(dataset_dir, split='train'):
    """Analyze image dimensions in the dataset"""
    
    dataset_path = Path(dataset_dir)
    split_path = dataset_path / split
    
    if not split_path.exists():
        print(f"Error: {split_path} not found")
        return
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(split_path.glob(f'*{ext}'))
        image_files.extend(split_path.glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images in '{split}' split")
    print("\nAnalyzing image dimensions...\n")
    
    widths = []
    heights = []
    aspect_ratios = []
    sizes = []
    
    # Sample images (check first 500 for speed)
    sample_size = min(500, len(image_files))
    
    for img_path in image_files[:sample_size]:
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
            sizes.append((w, h))
    
    # Statistics
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)
    
    print(f"Analyzed {len(widths)} images\n")
    
    print("Width statistics:")
    print(f"  Min: {widths.min()}")
    print(f"  Max: {widths.max()}")
    print(f"  Mean: {widths.mean():.1f}")
    print(f"  Median: {np.median(widths):.1f}")
    
    print("\nHeight statistics:")
    print(f"  Min: {heights.min()}")
    print(f"  Max: {heights.max()}")
    print(f"  Mean: {heights.mean():.1f}")
    print(f"  Median: {np.median(heights):.1f}")
    
    print("\nAspect ratio (width/height):")
    print(f"  Min: {aspect_ratios.min():.2f}")
    print(f"  Max: {aspect_ratios.max():.2f}")
    print(f"  Mean: {aspect_ratios.mean():.2f}")
    print(f"  Median: {np.median(aspect_ratios):.2f}")
    
    print("\nMost common dimensions:")
    size_counter = Counter(sizes)
    for size, count in size_counter.most_common(10):
        print(f"  {size[0]}x{size[1]}: {count} images ({count/len(sizes)*100:.1f}%)")
    
    # Recommendations
    print("\n" + "="*50)
    print("RECOMMENDATIONS:")
    print("="*50)
    
    avg_w = widths.mean()
    avg_h = heights.mean()
    avg_aspect = aspect_ratios.mean()
    
    # Suggest square sizes based on average
    avg_size = (avg_w + avg_h) / 2
    
    if avg_size < 150:
        suggested_sizes = [64, 128]
    elif avg_size < 300:
        suggested_sizes = [128, 256]
    elif avg_size < 600:
        suggested_sizes = [256, 512]
    else:
        suggested_sizes = [512, 1024]
    
    print(f"\nAverage image size: {avg_w:.0f}x{avg_h:.0f}")
    print(f"Average aspect ratio: {avg_aspect:.2f} ({'landscape' if avg_aspect > 1 else 'portrait' if avg_aspect < 1 else 'square'})")
    
    print(f"\nSuggested target sizes (square) for HOG:")
    for size in suggested_sizes:
        print(f"  {size}x{size} - {'Good for speed' if size <= 128 else 'Better quality, slower'}")
    
    print(f"\nSuggested target size (preserving aspect ratio):")
    if avg_aspect > 1:
        # Landscape
        for size in suggested_sizes:
            h = int(size / avg_aspect)
            print(f"  {size}x{h}")
    else:
        # Portrait
        for size in suggested_sizes:
            w = int(size * avg_aspect)
            print(f"  {w}x{size}")
    
    print("\nNote: Square resizing is recommended for HOG to maintain consistency.")
    print("Recommended: 128x128 (fast) or 256x256 (better quality)")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze image sizes in dataset')
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'test', 'valid'],
        help='Which split to analyze'
    )
    
    args = parser.parse_args()
    analyze_image_sizes(args.dataset_dir, args.split)
