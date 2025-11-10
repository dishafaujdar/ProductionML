# Alternative: Use sklearn's make_classification for a large synthetic dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt
from Logistic import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate large synthetic dataset (100,000 samples, 20 features)
X, y = make_classification(
    n_samples=100000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42,
    class_sep=0.8
)

print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression('synthetic_large', task='binary')
X_train, y_train = model.CleanData(X_train, y_train, remove_outliers=False)
X_test = model.scaler.transform(X_test)
model.fit(X_train, y_train, use_ridge=True)
model.test(X_test,y_test,show_confusion_matrix=True)
model.evaluate(X_train,y_train,X_test,y_test,show_confusion_matrix=True)


