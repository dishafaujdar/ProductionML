from pandas.core.api import unique
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class LogisticRegression():

  def __init__ (self, dataset_name, task='binary'):
      self.l_r = 0.01
      self.lambda_ = 0.001
      self.epochs = 5000
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
        
    if self.weight is not None:
      loss += (self.lambda_ / (2 * m)) * np.sum(self.weight ** 2)
        
    return loss

  def fit(self,X_train, y_train, use_ridge = False):
    self.lambda_ = self.lambda_ if use_ridge is True else 0

    m,n = X_train.shape
    self.bias = 0
    self.weight = np.zeros(n)

    if self.task == 'binary':
        for e in range(self.epochs):
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
        self.bias = np.zeros(n_classes)
        
        for e in range(self.epochs):
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
    if self.task == 'binary':
        if self.weight.ndim > 1:
            self.weight = self.weight[:, 0] if self.weight.shape[1] == 1 else self.weight.flatten()[:X.shape[1]]
        
        if isinstance(self.bias, np.ndarray) and self.bias.ndim > 0:
            self.bias = float(self.bias[0]) if self.bias.size > 0 else 0.0
        elif not isinstance(self.bias, (int, float)):
            self.bias = 0.0
        
        z = X @ self.weight + self.bias
        return self.sigmoid(z)
    else:
        z = X @ self.weight + self.bias
        return self.softmax(z)

  def predict(self, X, threshold=0.5):
        if self.task == 'binary':
            proba = self.predict_proba(X)
            if proba.ndim > 1:
                proba = proba.flatten()
            return (proba >= threshold).astype(int)
        else:
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)

  def test(self, X_test, y_test, show_confusion_matrix=True, save_plot=True, plot_filename=None):
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
            
            # Print confusion matrix in text format
            cm = confusion_matrix(y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(f"                Predicted")
            print(f"              Class 0  Class 1")
            print(f"Actual Class 0   {cm[0,0]:5d}   {cm[0,1]:5d}")
            print(f"        Class 1   {cm[1,0]:5d}   {cm[1,1]:5d}")
            print(f"\nTrue Negatives (TN):  {tn}")
            print(f"False Positives (FP): {fp}")
            print(f"False Negatives (FN): {fn}")
            print(f"True Positives (TP):  {tp}")
            
            # Visualize confusion matrix
            if show_confusion_matrix:
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Class 0', 'Class 1'],
                            yticklabels=['Class 0', 'Class 1'])
                plt.title("Confusion Matrix")
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                plt.tight_layout()
                
                if save_plot:
                    if plot_filename is None:
                        plot_filename = 'confusion_matrix.png'
                    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                    print(f"\n✓ Confusion matrix plot saved as '{plot_filename}'")
                
                try:
                    plt.show()
                except:
                    plt.close()
        
        # Visualize predictions vs actual (probability distribution)
        proba = self.predict_proba(X_test)
        if self.task == 'binary':
            plt.figure(figsize=(10, 5))
            plt.hist(proba[y_test == 0], bins=30, alpha=0.5, label='Class 0')
            plt.hist(proba[y_test == 1], bins=30, alpha=0.5, label='Class 1')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Frequency')
            plt.legend()
            plt.title('Predicted Probability Distribution')
            
            if save_plot:
                prob_filename = plot_filename.replace('.png', '_probabilities.png') if plot_filename else 'probability_distribution.png'
                plt.savefig(prob_filename, dpi=150, bbox_inches='tight')
            
            try:
                plt.show()
            except:
                plt.close()

  def evaluate(self, X_train, y_train, X_test, y_test, show_confusion_matrix=True, save_plots=True):

        """
        Evaluate model on both training and test sets with confusion matrices.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        show_confusion_matrix : bool, default=True
            Whether to display confusion matrix plots
        save_plots : bool, default=True
            Whether to save confusion matrix plots as PNG files
        """
        print("\n" + "="*50)
        print("=== Training Set Performance ===")
        print("="*50)
        self.test(X_train, y_train, 
                  show_confusion_matrix=show_confusion_matrix,
                  save_plot=save_plots,
                  plot_filename='confusion_matrix_train.png')
        
        print("\n" + "="*50)
        print("=== Test Set Performance ===")
        print("="*50)
        self.test(X_test, y_test,
                  show_confusion_matrix=show_confusion_matrix,
                  save_plot=save_plots,
                  plot_filename='confusion_matrix_test.png')
        
        # Compare training vs test accuracy
        y_pred_train = self.predict(X_train)
        y_pred_test = self.predict(X_test)
        
        train_acc = np.mean(y_pred_train == y_train)
        test_acc = np.mean(y_pred_test == y_test)
        
        print(f"\n{'='*50}")
        print("=== Model Generalization Analysis ===")
        print(f"{'='*50}")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:     {test_acc:.4f}")
        print(f"Difference:        {abs(train_acc - test_acc):.4f}")
        
        if abs(train_acc - test_acc) > 0.05:
            print("⚠️  Warning: Large gap suggests possible overfitting!")
        else:
            print("✓ Model generalizes well (small gap between train/test)")
        print("="*50)