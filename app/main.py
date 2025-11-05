# linear regression from scratch
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


class MyLinearRegression():
  def __init__(self, dataset_name):
    self.X = None
    self.y = None
    self.Ridge = None
    self.scaler = None
    self.epochs = 10000
    self.learning_rate = 0.01
    self.weights = None
    self.lambda_ = 0.01
    self.dataset_name = dataset_name
    self.X_train = None
    self.y_train = None
    self.X_test = None
    self.y_test = None

  def CleanData(self, X_train, y_train, remove_outliers=False):
    if remove_outliers:
      # IQR logic here, but default to False
      Q1 = np.percentile(y_train,25)
      Q3 = np.percentile(y_train,75)
      IQR = Q3 - Q1
      mask = (y_train >= Q1 - 1.5*IQR) & (y_train <= Q3 + 1.5*IQR)
      X_train = X_train[mask]
      y_train = y_train[mask]
    
    self.scaler = StandardScaler()
    X_train = self.scaler.fit_transform(X_train)
    
    return X_train,y_train

  def Visuals(self,X_train,y_train,plot_types = 'boxplot'):
    if plot_types == 'boxplot':
      sns.boxplot(data = y_train)
      plt.title(f'{self.dataset_name} - Target Distribution (Boxplot)')
      plt.xlabel('Target Variable')
      plt.show()
    elif plot_types == 'histogram':
      sns.histplot(data = y_train)
      plt.title(f'{self.dataset_name} - Target Distribution (Histogram)')
      plt.xlabel('Target Variable')
      plt.show()
    elif plot_types == 'scatter':
      sns.scatterplot(data = y_train)
      plt.title(f'{self.dataset_name} - Target Distribution (Scatter)')
      plt.xlabel('Target Variable')
      plt.ylabel('Target Variable')
      plt.show()

  def fit(self,X_train,y_train,use_ridge = False, lambda_=None, method = 'auto'):
    if method == "auto":
      method = 'normal_eq' if X_train.shape[0] < 20000 else 'gradient_descent'

    if method == 'normal_eq':
      self.weights = self._normal_equation(X_train, y_train, use_ridge, self.lambda_)
    elif method == 'gradient_descent':
        self.weights = self._gradient_descent(X_train, y_train, use_ridge, self.lambda_)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return self

  def _normal_equation(self, X, y, use_ridge, lambda_):

    X_with_bias = np.column_stack([np.ones((X.shape[0])), X])

    XTX = X_with_bias.T @ X_with_bias
    XTy = X_with_bias.T @ y

    print(f"XTX shape: {XTX.shape}")
    print(f"XTy shape: {XTy.shape}")

    if XTy.ndim > 1:
        XTy = XTy.flatten()
    elif XTy.ndim == 0:
        XTy = np.array([XTy])

    if (use_ridge):
        XTX += lambda_ * np.eye(X_with_bias.shape[1])
        XTX[0,0] -= lambda_
    else:
        XTX += 1e-10 * np.eye(X_with_bias.shape[1])
    
    w = np.linalg.solve(XTX, XTy)
    if w.ndim == 0:
      w = np.array([w])
    print(f"Weights shape: {w.shape}")
    return w

  def _gradient_descent(self, X, y, use_ridge, lambda_):
    X_with_bias = np.column_stack([np.ones((X.shape[0])), X])
    self.weights = np.zeros(X_with_bias.shape[1])
    
    for epoch in range(self.epochs):
      y_pred = X_with_bias @ self.weights
      error = y_pred - y
      if use_ridge:
        gradient = X_with_bias.T @ error / X_with_bias.shape[0] + (lambda_ / X_with_bias.shape[0]) * self.weights
        gradient[1:] += (lambda_ / X_with_bias.shape[0]) * self.weights[1:]  # Regularize only non-bias weights
      else:
        gradient = X_with_bias.T @ error / X_with_bias.shape[0]
      self.weights -= self.learning_rate * gradient
    return self.weights

  def predict(self,X):
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    return X_with_bias @ self.weights

  def test(self,X_test,y_test):

    y_pred = self.predict(X_test)
    
    mse = np.mean((y_test - y_pred)**2)
    r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
    plt.title(f'{self.dataset_name} - Predictions vs Actual (R² = {r2:.3f})')

    print(f"MSE: {mse}, R²: {r2}")

    plt.title(f'{self.dataset_name} - Residual Plot')
    # Visualize predictions vs actual
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"R² = {r2:.3f}")
    plt.show()
  
    # Visualize residuals
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.show()