from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def run_pca(X):
    # standardize the data before applying PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize PCA, specifying the number of components you want to keep
    pca = PCA(n_components=2)

    # Fit PCA on the dataset and transform the data
    X_pca = pca.fit_transform(X_scaled)

    return X_pca

# print("Original shape:", X.shape)
# print("Reduced shape:", X_pca.shape)

# # Access the explained variance
# print("Explained variance ratio:", pca.explained_variance_ratio_)
