from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D 
# Load data from CSV
df = pd.read_csv('experiments/inference/triplet_embed2D.csv')

# Extract features and labels
features = df[['x1', 'x2']].values
labels = df['label'].values

# Standardize features if needed (optional)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Create t-SNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=700)
tsne_results = tsne.fit_transform(features_scaled)

# Plot t-SNE in 2D with color representing the third dimension (label)
plt.figure(figsize=(10, 8))

# Create a colormap based on labels
cmap = plt.cm.get_cmap('tab10')  # Adjust colormap if desired
colors = cmap(labels)

# Scatter plot in 2D with color representing the third dimension (label)
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=cmap)

# Add colorbar to indicate the correspondence between colors and labels
cbar = plt.colorbar()
cbar.set_label('Label')

plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title("t-SNE 2D Scatter Plot with Color Representing Labels")
plt.show()

# Perform k-means clustering on t-SNE embeddings
kmeans = KMeans(n_clusters=10, random_state=42)  # Assuming 10 clusters based on the number of unique labels
kmeans.fit(tsne_results)

# Get cluster labels
cluster_labels = kmeans.labels_

# Calculate accuracy
kmeans_accuracy = accuracy_score(labels, cluster_labels)
print("K-Means Accuracy:", kmeans_accuracy)
