#%% Imports 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#%% Set Number of components for PCA
N_COMPONENTS = 500

#%% Function to perform PCA on dataset
def perform_pca(data_test, data_train):
    scaler = StandardScaler()
    new_data_train = scaler.fit_transform(data_train)

    pca = PCA(n_components=N_COMPONENTS)
    transformed_data_train = pca.fit_transform(new_data_train)

    new_data_test = scaler.transform(data_test)
    transformed_data_test = pca.transform(new_data_test)

    return transformed_data_train, transformed_data_test