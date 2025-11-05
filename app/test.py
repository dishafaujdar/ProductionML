from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from main import MyLinearRegression
# from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes

# 1. Load data
data = load_diabetes()
X, y = data.data, data.target

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize model
model = MyLinearRegression(dataset_name="Diabetes")
# 4. Clean training data
X_train, y_train = model.CleanData(X_train, y_train, remove_outliers=False)

model.fit(X_train, y_train, use_ridge=True, lambda_=0.1)

# 5. Scale test data using training scaler
X_test = model.scaler.transform(X_test)

# 6. Fit model

# 7. Test
model.test(X_test, y_test)