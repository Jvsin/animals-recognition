#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
from scripts.dataset.unpack_root_dataset import main_unpack
from scripts.pca import perform_pca
from scripts.dataset.create_splits import create_splits, convert_class_name_to_index
from scripts.scaler import rescale_all_images, INPUT_DIR
from scripts.classifiers.svm import SVMClassifier
from scripts.classifiers.random_forests import RandomForestClassifier

#%% Extract dataset paths and labels from a given directory
main_unpack()
create_splits()

#%% Resize and pad images to a fixed size
rescale_all_images()

#%% Use extraction and resizing functions to prepare dataset
x_hog_train = []
x_hog_test = []
# todo: other feature lists

y_train = []
y_test = []

data_train = pd.read_csv(f'{INPUT_DIR}/train.csv')
y_train = list(map(convert_class_name_to_index, data_train['class'].tolist()))

for _, row in data_train.iterrows():
    image_path = f"{INPUT_DIR}/{row['image_path']}"
    image = plt.imread(image_path)

            # todo: Hog feature extraction
            # todo: Other feature extraction methods

            # todo: save extracted features

data_test = pd.read_csv(f'{INPUT_DIR}/test.csv')
y_test = list(map(convert_class_name_to_index, data_test['class'].tolist()))

for _, row in data_test.iterrows():
    image_path = f"{INPUT_DIR}/{row['image_path']}"
    image = plt.imread(image_path)

            # todo: Hog feature extraction
            # todo: Other feature extraction methods

            # todo: save extracted features


#%% PCA 
x_hog_train_pca, x_hog_test_pca = perform_pca(x_hog_test, x_hog_train)
# todo: the same for other features methods

#%% Classification - SVM
svm_hog_classifier = SVMClassifier()
svm_hog_classifier.fit(x_hog_train_pca, y_train)
y_hog_pred = svm_hog_classifier.predict(x_hog_test_pca)
# todo: the same for other features methods

#%% Classification - Random Forest
rf_hog_classifier = RandomForestClassifier()
rf_hog_classifier.fit(x_hog_train_pca, y_train)
y_hog_pred_rf = rf_hog_classifier.predict(x_hog_test_pca)
# todo: the same for other features methods


#%% Metrics 
# todo: print all metrics from scripts.classifiers.metrics for each classifier and feature method
