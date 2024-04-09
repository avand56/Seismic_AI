from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Example dataset with 100 samples and 5 features
X = np.random.rand(100, 5)

# It's a good practice to standardize the data before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize PCA, specifying the number of components you want to keep
pca = PCA(n_components=2)

# Fit PCA on the dataset and transform the data
X_pca = pca.fit_transform(X_scaled)

print("Original shape:", X.shape)
print("Reduced shape:", X_pca.shape)

# Access the explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)
