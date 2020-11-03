from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np

import matplotlib.pyplot as plt

iris = datasets.load_iris()
print(list(iris.keys()))
print(iris.feature_names)  # 3, 4th
print(iris.target_names)  # 2, 3rd
X = iris["data"][:, 3:]  # petal width (cm)
y = (iris["target"] == 2).astype(np.int)  # is 'virginica'
print(X)
print(y)

regression1 = LogisticRegression()
regression1.fit(X, y)
print(regression1.coef_)
print(regression1.intercept_)
