# kod z deep learning fundamentals
import os
import csv
import random
from os.path import join
from pathlib import Path

random.seed(42)

final_classes = {
    'cheetah': 0,
    'elephant': 1,
    'giraffe': 2,
    'lion': 3,
    'rhino': 4,
    'zebra': 5,
}

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
dataset_root = str(project_root / 'dataset')

train_ratio = 0.7
test_ratio = 0.3


def convert_class_name_to_index(class_name):
    return final_classes.get(class_name, -1)

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
        for split_dir in ['train', 'test', 'valid']:
            split_path = join(comb_root, split_dir)
            if not os.path.exists(split_path):
                print(f"Warning: Split directory {split_path} does not exist. Skipping.")
                continue
            
            class_images = {cls: [] for cls in classes.keys()}
            
            for f in os.listdir(split_path):
                full_path = join(split_path, f)
                if os.path.isfile(full_path) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    filename_lower = f.lower()
                    for cls in classes.keys():
                        if filename_lower.startswith(cls.lower()):
                            rel_path = os.path.relpath(full_path, comb_root)
                            class_images[cls].append(rel_path)
                            break
            
            for cls, images in class_images.items():
                if not images:
                    continue
                
                target_split = 'test' if split_dir == 'valid' else split_dir
                
                if target_split in writers:
                    for img in images:
                        writers[target_split].writerow([img, cls])
                
                print(f"Split {split_dir} - Class {cls}: {len(images)} images -> {target_split}.csv")
    
    finally:
        for f in csv_files.values():
            f.close()

def create_splits(force_extract=False):
    if not force_extract and (Path(dataset_root) / 'train.csv').exists() and (Path(dataset_root) / 'test.csv').exists():
        print("CSV splits already exist. Skipping creation.")
        return
    
    create_csv_files(dataset_root)
    
    split_and_write_csvs(dataset_root, dataset_root, final_classes,
                         train_ratio)


if __name__ == "__main__":
    create_splits()
    
    print("Dataset splitting completed!")