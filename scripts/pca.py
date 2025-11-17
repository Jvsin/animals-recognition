#%% Imports 
from sklearn.decomposition import PCA

#%% Set Number of components for PCA
N_COMPONENTS = 500

#%% Function to perform PCA on dataset
def perform_pca(data):
    pca = PCA(n_components=N_COMPONENTS)
    transformed_data = pca.fit_transform(data)
    return transformed_data, pca
