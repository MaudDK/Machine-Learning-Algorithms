from sklearn.model_selection import train_test_split
from sklearn import datasets
from models.unsupervised.kmeans import KMeans
from utils.plotting import plot_cluster_test

#Generate Clusters Dataset
X, y = datasets.make_blobs(n_samples= 100, centers= 4, n_features= 2, random_state=42)

#Modeling
model = KMeans(k=4)
model.fit(X, max_iters=100)

#Predictions
y_pred = model.predict(X)

#Plotting
plot_cluster_test(X, y, y_pred, model.centroids)