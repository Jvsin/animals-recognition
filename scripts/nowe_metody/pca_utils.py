from sklearn.decomposition import PCA
import numpy as np

#cel działać na różnych zestawach cech

def dopasuj_pca_nkomp(X_train, n_components=64, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    return pca, X_train_pca

def dopasuj_pca_wariancja(X_train, explained_variance=0.95, random_state=42):
    pca = PCA(n_components=explained_variance, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    return pca, X_train_pca

def przeksztalc_pca(pca, X):
    return pca.transform(X)

def policz_skumulowana_wariancje(X_train, max_components=200, random_state=42):
    pca = PCA(n_components=max_components, random_state=random_state)
    pca.fit(X_train)
    return np.cumsum(pca.explained_variance_ratio_)
