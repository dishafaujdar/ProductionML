from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from main import MyLinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes


data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MyLinearRegression(dataset_name="California Housing")

X_train, y_train = model.CleanData(X_train, y_train, remove_outliers=False)

model.fit(X_train, y_train, use_ridge=False, lambda_=None)

X_test = model.scaler.transform(X_test)

model.test(X_test, y_test)