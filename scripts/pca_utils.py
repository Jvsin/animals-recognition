from sklearn.decomposition import PCA
import numpy as np

def dopasuj_pca_nkomp(X_train, n_components=64, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    return pca, X_train_pca

def przeksztalc_pca(pca, X):
    return pca.transform(X)