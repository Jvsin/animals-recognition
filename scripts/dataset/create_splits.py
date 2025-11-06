import os
import csv
import random
from os.path import join
from pathlib import Path

random.seed(42)

final_classes = {
    'Cat': 0,
    'Cow': 1,
    'Deer': 2,
    'Dog': 3,
    'Goat': 4,
    'Hen': 5,
    'Rabbit': 6,
    'Sheep': 7,
}

# Calculate dataset_root as absolute path from project root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
dataset_root = str(project_root / 'dataset')

# Split ratios
train_ratio = 0.7
test_ratio = 0.3


def create_csv_files(out_root):
    os.makedirs(out_root, exist_ok=True)
    for split in ['train', 'test']:
        csv_path = join(out_root, f'{split}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'class'])


def split_and_write_csvs(comb_root, out_root, classes, train_r):
    csv_files = {}
    for split in ['train', 'test']:
        csv_path = join(out_root, f'{split}.csv')
        csv_files[split] = open(csv_path, 'a', newline='', encoding='utf-8')
    
    writers = {split: csv.writer(f) for split, f in csv_files.items()}
    
    try:
        for cls in classes.keys():
            cls_dir = join(comb_root, cls)
            if not os.path.exists(cls_dir):
                print(f"Warning: Class directory {cls_dir} does not exist. "
                      "Skipping.")
                continue
            
            # Look for images in <cls>/train/images/ directory
            train_images_dir = join(cls_dir, 'train', 'images')
            images = []
            
            if os.path.exists(train_images_dir):
                for f in os.listdir(train_images_dir):
                    full_path = join(train_images_dir, f)
                    if os.path.isfile(full_path) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        # Store relative path from dataset root
                        rel_path = os.path.relpath(full_path, comb_root)
                        images.append(rel_path)
            
            if not images:
                print(f"Warning: No images found in {train_images_dir}. Skipping.")
                continue
            
            # Shuffle the images
            random.shuffle(images)
            n = len(images)
            
            # Calculate split indices
            train_end = int(train_r * n)
            
            train_imgs = images[:train_end]
            test_imgs = images[train_end:]
            # Write to CSVs
            splits_data = [('train', train_imgs), ('test', test_imgs)]
            for split, imgs in splits_data:
                for img in imgs:
                    writers[split].writerow([img, cls])
            
            print(f"Class {cls}: {len(train_imgs)} train, "
                  f"{len(test_imgs)} test images.")
    finally:
        for f in csv_files.values():
            f.close()


if __name__ == "__main__":
    # Create CSV files
    create_csv_files(dataset_root)
    
    # Split and write
    split_and_write_csvs(dataset_root, dataset_root, final_classes,
                         train_ratio)
    
    print("Dataset splitting completed!")