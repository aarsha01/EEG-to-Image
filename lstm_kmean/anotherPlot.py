import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # Importing 3D plotting tools
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Load data from CSV
df = pd.read_csv('experiments/inference/triplet_embed2D.csv')

# Extract features and labels
features = df[['x1', 'x2']].values
labels = df['label'].values

# Standardize features if needed (optional)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Create Isomap with 2 components (2D)
isomap = Isomap(n_neighbors=28, n_components=2)  # Adjust hyperparameters as needed
isomap_results = isomap.fit_transform(features)

# Plot Isomap in 2D with color representing the third dimension (label)
plt.figure(figsize=(10, 8))

# Create a colormap based on labels
cmap = plt.cm.get_cmap('tab10')  # Adjust colormap if desired
colors = cmap(labels)

# Scatter plot in 2D with color representing the third dimension (label)
plt.scatter(isomap_results[:, 0], isomap_results[:, 1], c=labels, cmap=cmap)

# Add colorbar to indicate the correspondence between colors and labels
cbar = plt.colorbar()
cbar.set_label('Label')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title("Isomap 2D Scatter Plot with Color Representing Labels")
plt.show()

# Perform k-means clustering
kmeans = KMeans(n_clusters=10, random_state=42)  # Assuming 10 clusters based on the number of unique labels
kmeans.fit(features_scaled)  # Using standardized features if applicable

# Get cluster labels
cluster_labels = kmeans.labels_

# Map cluster labels to original labels
label_mapping = {cluster: np.argmax(np.bincount(labels[cluster_labels == cluster])) for cluster in range(10)}
mapped_cluster_labels = np.array([label_mapping[cluster] for cluster in cluster_labels])

# Calculate accuracy
kmeans_accuracy = accuracy_score(labels, mapped_cluster_labels)
print("K-Means Accuracy:", kmeans_accuracy)