from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from models.linear_models.ElasticNet import ElasticNet
from metrics.loss import rmse, mean_squared_error, mean_absolute_loss

#Generate Regression Dataset
X, y = datasets.make_regression(n_samples= 100, n_features=1, noise = 20, random_state= 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Train
linreg = ElasticNet(n_iters=1000, learning_rate = 0.01)
linreg.fit(X_train, y_train)

#Predictions
y_train_preds = linreg.predict(X_train)
y_test_preds = linreg.predict(X_test)
y_pred_line = linreg.predict(X)

#Metrics
mse = mean_squared_error(y_test, y_test_preds)
mae = mean_absolute_loss(y_test, y_test_preds)
rmse = rmse(y_test, y_test_preds)
print(f'Test: MAE(L1):{mae} | MSE(L2): {mse} | RMSE: {rmse}')

#Plot
fig = plt.figure(figsize=(8,6))
plt.scatter(X_train[:,0], y_train, color = "g", s = 10)
plt.scatter(X_test[:,0], y_test, color = "r", s = 10)
plt.plot(X[:,0], y_pred_line, color='red', linewidth=1)
plt.show()

print(linreg.W, linreg.b)





