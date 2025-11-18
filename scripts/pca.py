#%% Imports
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#%% Set Number of components for PCA
N_COMPONENTS = 100

#%% Function to perform PCA on dataset
def perform_pca(data_test, data_train):
    data_train = np.asarray(data_train)
    data_test = np.asarray(data_test)

    # sanity check
    if data_train.shape[1] != data_test.shape[1]:
        raise ValueError(
            f"Train and test must have the same number of features"
        )

    scaler = StandardScaler()
    new_data_train = scaler.fit_transform(data_train)

    max_components = min(new_data_train.shape[0], new_data_train.shape[1])
    n_components = min(N_COMPONENTS, max_components)

    if n_components <= 0:
        raise ValueError(
            f"Cannot perform PCA: computed n_components={n_components} "
            f"from N_COMPONENTS={N_COMPONENTS}, "
            f"n_samples={new_data_train.shape[0]}, "
            f"n_features={new_data_train.shape[1]}"
        )

    pca = PCA(n_components=n_components)
    transformed_data_train = pca.fit_transform(new_data_train)

    new_data_test = scaler.transform(data_test)
    transformed_data_test = pca.transform(new_data_test)

    return transformed_data_train, transformed_data_test