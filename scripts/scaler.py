#%% Imports 
from skimage.transform import resize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#%% Set size of image 
NEW_IMAGE_SIZE = (256, 256)

#%% Set Directories
INPUT_DIR = 'dataset/'
OUTPUT_DIR = 'output/images/'

#%% Function to paint image to square by adding padding
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
    

#%% Function to rescale image to NEW_IMAGE_SIZE
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

#%% rescale all images in a given directory and save to output directory
def rescale_all_images():
    print("Rescaling all images...")
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
    
    print("Rescaling completed.")
