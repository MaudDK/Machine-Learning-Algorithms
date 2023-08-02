from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from models.classification.LogisticRegression import LogisticRegression
from metrics.loss import binary_cross_entropy

import numpy as np

#Generate classification dataset
ds= datasets.load_breast_cancer()
X, y = ds.data, ds.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Train
logreg = LogisticRegression(n_iters=1000, learning_rate = 0.01)
logreg.fit(X_train, y_train)

#Predict
y_train_preds = logreg.predict(X_train)
y_test_preds = logreg.predict(X_test)
y_pred_line = logreg.predict(X)

#Metrics
logloss = binary_cross_entropy(y_test, y_test_preds)
print(f'Logistic Loss: {logloss}')

# Plot the dataset
fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Classification Dataset')
plt.show()

