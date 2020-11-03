# 將feature變為多項式

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

x = np.array([[5], [15], [25], [35], [45], [55]])
y = np.array([15, 11, 2, 8, 25, 32])
plt.plot(x, y)
plt.scatter(x, y)
regression1 = LinearRegression()
regression1.fit(x, y)
x_sequence = np.array(np.arange(5, 55, 0.1)).reshape((-1, 1))
plt.plot(x, regression1.coef_ * x + regression1.intercept_)
score1 = regression1.score(x, y)
print("linear regression score={}".format(score1))

transformer = PolynomialFeatures(degree=2, include_bias=False)
"""
PolynomialFeatures: 將feature變為多項式
etc: y=ax+b -> y=(a_1)x + (a_2)x^2 + b
degree: 級數
include_bias: 是否包含常數項
"""
transformer.fit(x)
x_ = transformer.transform(x)
"""
x_: 轉換後的x
x: [[5], [10], [15]...]
x_: [[5, 25], [10, 100], [15, 225]...]
"""
print(f"x shape={x.shape},(x**2)x_ shape={x_.shape}")
regression2 = LinearRegression().fit(x_, y)
score2 = regression2.score(x_, y)
print("2nd order regression score={}".format(score2))
print("2nd order coef={}".format(regression2.coef_))
print("2nd order intercept={}".format(regression2.intercept_))
x_sequence_ = transformer.transform(x_sequence)
y_predict = regression2.predict(x_sequence_)
plt.plot(x_sequence, y_predict, "--")
plt.show()
