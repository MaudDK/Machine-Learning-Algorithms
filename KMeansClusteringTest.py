from sklearn.model_selection import train_test_split
from sklearn import datasets
from models.unsupervised.kmeans import KMeans
from utils.plotting import plot_cluster_test

#Generate Clusters Dataset
X, y = datasets.make_blobs(n_samples= 100, centers= 3, n_features= 2)

#Modeling
model = KMeans(k=3)
model.fit(X, max_iters=1000)

#Predictions
y_pred = model.predict(X)

#Plotting
plot_cluster_test(X, y, y_pred, model.centroids)