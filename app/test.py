from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from Linear import MyLinearRegression
from Logistic import LogisticRegression
from sklearn.datasets import load_breast_cancer

# For Logistic Regression - use breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression('breast_cancer', task='binary')
X_train, y_train = model.CleanData(X_train, y_train, remove_outliers=False)
X_test = model.scaler.transform(X_test)
model.fit(X_train, y_train, use_ridge=True)
model.test(X_test,y_test,show_confusion_matrix=True)
model.evaluate(X_train,y_train,X_test,y_test,show_confusion_matrix=True)

# data = fetch_california_housing()
# X, y = data.data, data.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = MyLinearRegression(dataset_name="California Housing")
# X_train, y_train = model.CleanData(X_train, y_train, remove_outliers=False)
# X_test = model.scaler.transform(X_test)
# model.fit(X_train, y_train, use_ridge=False, lambda_=None)
# model.test(X_test, y_test)