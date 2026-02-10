from skimage.transform import resize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

NEW_IMAGE_SIZE = (256, 256)
INPUT_DIR = 'dataset/'
OUTPUT_DIR = 'output/images/'

#Funkcja do dopasowania obrazu poprzez dodanie marginesów (paddingu).
def pad_to_square(image, old_shape, new_shape):
    diff_y = new_shape[0] - old_shape[0] 
    diff_x = new_shape[1] - old_shape[1]

    pad_top = diff_y // 2
    pad_bottom = diff_y - pad_top

    pad_left = diff_x // 2
    pad_right = diff_x - pad_left

    color = np.median(image, axis=(0,1))
    pad = ((pad_top, pad_bottom), (pad_left, pad_right))
    padded = np.stack([np.pad(image[:,:,c], pad, mode='constant', constant_values=color[c]) for c in range(3)], axis=2)
   
    return padded
    
#Funkcja do przeskalowania obrazu do rozmiaru NEW_IMAGE_SIZE.
def rescale_image(image):
    shapes = list(image.shape)  
    new_shape = list(image.shape)
    if shapes[0] > shapes[1]:
        new_shape[1] = new_shape[0]
    elif shapes[1] > shapes[0]:
        new_shape[0] = new_shape[1]

    squared_image = pad_to_square(image, shapes, new_shape)
    rescaled_image = resize(squared_image, NEW_IMAGE_SIZE, anti_aliasing=True)

    return rescaled_image

#Przeskaluj wszystkie obrazy w podanym katalogu i zapisz je w katalogu wyjściowym.
def rescale_all_images(force_extract=False):
    output_dir = Path(OUTPUT_DIR)
    if not force_extract and all((output_dir / folder).exists() and any((output_dir / folder).iterdir()) for folder in ['train', 'test', 'valid']):
        print("Przeskalowany dataset już istnieje i nie jest pusty. Pomijam rozpakowywanie.")
        return

    print("skalowanie wszystkich obrazów...")
    for name in ['train', 'test']:
        print(f"Processing {name} set...")
        data = pd.read_csv(f'{INPUT_DIR}/{name}.csv')
        for _, row in data.iterrows():
            image_path = f"{INPUT_DIR}/{row['image_path']}"
            image = plt.imread(image_path)
            rescaled_image = rescale_image(image)
            output_path = f"{OUTPUT_DIR}/{row['image_path']}"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.imsave(output_path, rescaled_image)
    
    print("Przeskalowywanie zakończone")
