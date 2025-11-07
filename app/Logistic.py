from pandas.core.api import unique
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression(): 
  def __init__ (self, dataset_name, task='binary'):
      self.l_r = 0.01
      self.lambda_ = 0.001
      self.epochs = 1000
      self.task = task
      self.scaler = None
      self.weight = None
      self.bias = None
      self.Ridge = None

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

  def Visuals(self,y_train,plot_types = 'boxplot'):
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

  def sigmoid(self,z):
    z = np.clip(z, -500,500)
    return 1 / (1 + np.exp(-z))

  def softmax(self,z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

  def binary_ce(self,y_true,y_pred):
    m = y_true.shape[0]
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
    # Add L2 regularization
    if self.weight is not None:
      loss += (self.lambda_ / (2 * m)) * np.sum(self.weight ** 2)
        
    return loss
  
  def multinomial_ce(self,y_true,y_pred):
    m = y_true.shape[0]
    y_pred = np.clip(y_pred, 1e-7, 1)
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
    if self.weights is not None:
      loss += (self.lambda_ / (2 * m)) * np.sum(self.weights ** 2)
        
    return loss

  def fit(self,X_train, y_train, use_ridge = False):
    self.lambda_ = self.lambda_ if use_ridge is True else 0

    m,n = X_train.shape
    self.bias = 0
    self.weight = np.zeros(n)

    for e in range(self.epochs):
        if self.task == 'binary':
            z = X_train @ self.weight + self.bias
            y_pred = self.sigmoid(z)

            dw = (1/m) * X_train.T @ (y_pred.reshape(-1,1) - y_train.reshape(-1, 1))
            db = (1/m) * np.sum(y_pred.reshape(-1,1) - y_train.reshape(-1, 1))
                        
            dw += (self.lambda_ / m) * self.weight.reshape(-1,1)
                        
            self.weight -= self.l_r * dw.flatten()
            self.bias -= self.l_r * db
                        
            if e % 100 == 0:
                loss = self.binary_ce(y_train.reshape(-1, 1), y_pred.reshape(-1,1))
                print(f"Epoch {e}, Loss: {loss:.4f}")

    else:
      
      n_classes = len(np.unique(y_train))
      self.weight = np.zeros((n, n_classes))
      z = X_train @ self.weight + self.bias
      y_pred = self.softmax(z)

      y_onehot = np.eye(self.weight.shape[1])[y_train.astype(int)]

      dw = (1/m) * X_train.T @ (y_pred - y_onehot)
      db = (1/m) * np.sum(y_pred - y_onehot, axis=0)
                
      dw += (self.lambda_ / m) * self.weight
                
      self.weight -= self.l_r * dw
      self.bias -= self.l_r * db 
                
      if e % 100 == 0:
        loss = self.multinomial_ce(y_onehot, y_pred)
        print(f"Epoch {e}, Loss: {loss:.4f}")
        
    return self

  def predict_proba(self,X):

    if self.task == 'binary' and self.weight.ndim > 1:
        self.weight = self.weight[:, 0] if self.weight.shape[1] == 1 else self.weight.flatten()[:X.shape[1]]

    if self.task == 'binary':
        if isinstance(self.bias, np.ndarray) and self.bias.ndim > 0:
            self.bias = float(self.bias[0]) if self.bias.size > 0 else 0.0
        elif not isinstance(self.bias, (int, float)):
            self.bias = 0.0
    
    print(f"DEBUG: X.shape={X.shape}, self.weight.shape={self.weight.shape}, self.bias.shape={getattr(self.bias, 'shape', 'scalar')}, self.task={self.task}")
    
    z = X @ self.weight + self.bias

  def predict(self, X, threshold=0.5):
        if self.task == 'binary':
            proba = self.predict_proba(X)
            if proba.ndim > 1:
               proba = proba.flatten()
            return (proba >= threshold).astype(int)

        else:
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
    
  def test(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        accuracy = np.mean(y_pred == y_test)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Confusion matrix for binary
        if self.task == 'binary':
            tn = np.sum((y_pred == 0) & (y_test == 0))
            fp = np.sum((y_pred == 1) & (y_test == 0))
            fn = np.sum((y_pred == 0) & (y_test == 1))
            tp = np.sum((y_pred == 1) & (y_test == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Visualize predictions vs actual
        proba = self.predict_proba(X_test)
        if self.task == 'binary':
            plt.hist(proba[y_test == 0], bins=30, alpha=0.5, label='Class 0')
            plt.hist(proba[y_test == 1], bins=30, alpha=0.5, label='Class 1')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()