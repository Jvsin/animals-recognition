#%% Imports 
from skimage.transform import resize
from pathlib import Path
from skimage.io import imread
import numpy as np

#%% Set size of image 
NEW_IMAGE_SIZE = (128, 128)

#%% Function to paint image to square by adding padding
def pad_to_square(image, old_shape, new_shape):
    diff_y = new_shape[0] - old_shape[0] 
    diff_x = new_shape[1] - old_shape[1]

    pad_top = diff_y // 2
    pad_bottom = diff_y - pad_top

    pad_left = diff_x // 2
    pad_right = diff_x - pad_left

    padded = np.pad(
        image,
        (
            (pad_top, pad_bottom),   
            (pad_left, pad_right),   
            (0, 0)                  
        ),
        mode="constant",
        constant_values=0
    )

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
    

#%% test 
# path to dataset/test/cheetah168250145_.jpg relative to this script
dataset_path = Path(__file__).resolve().parent.parent / "dataset" / "test" / "cheetah168250145_.jpg"

if not dataset_path.exists():
    raise FileNotFoundError(f"File not found: {dataset_path}")

image = imread(str(dataset_path))
rescaled_image = rescale_image(image)

import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4))
plt.imshow(rescaled_image)
plt.axis('off')
plt.title("Rescaled image")
plt.show()
    