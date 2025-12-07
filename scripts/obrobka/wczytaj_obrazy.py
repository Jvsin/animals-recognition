import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

CSV_DIR = Path("dataset")
IMG_DIR = Path("output/images") #katalog z obrazami po rescalingu

def load_image_paths(split="train", scaled=True):
    csv_path = CSV_DIR / f"{split}.csv"
    df = pd.read_csv(csv_path)
    image_paths = []
    base_dir = IMG_DIR if scaled else CSV_DIR

    for _, row in df.iterrows():
        image_path = base_dir / row["image_path"]
        image_paths.append(image_path)
    return image_paths

def load_images(split="train", scaled=True):
    paths = load_image_paths(split=split, scaled=scaled)
    images = []
    for p in paths:
        img = plt.imread(p)  #do numpya
        images.append(img)
    return images, paths
