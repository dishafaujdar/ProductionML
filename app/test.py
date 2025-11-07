from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from Linear import MyLinearRegression
from Logistic import LogisticRegression
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer


data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = MyLinearRegression(dataset_name="California Housing")
model = LogisticRegression(dataset_name='load_breast_caner',task='binary')
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression('breast_cancer', task='binary')
model.fit(X_train, y_train, use_ridge=True)
model.test(X_test, y_test)
# X_test = model.scaler.transform(X_test)

# X_train, y_train = model.CleanData(X_train, y_train, remove_outliers=False)
# model.fit(X_train, y_train, use_ridge=False, lambda_=None)
# model.test(X_test, y_test)