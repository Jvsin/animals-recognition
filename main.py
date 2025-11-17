#%% Imports
from scripts.dataset.unpack_root_dataset import main_unpack
from scripts.dataset.create_splits import create_splits
from scripts.scaler import rescale_all_images

#%% Extract dataset paths and labels from a given directory
main_unpack()
create_splits()

#%% Resize and pad images to a fixed size
rescale_all_images()

#%% Use extraction and resizing functions to prepare dataset


#%% PCA 


#%% Classification 


#%% Metrics 

