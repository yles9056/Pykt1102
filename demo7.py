"""
import matplotlib.pyplot as plt
from sklearn import linear_model

features = [[0, 1], [1, 3], [2, 6]]
values = [1, 4, 5.5]
regression = linear_model.LinearRegression()
regression.fit(features, values)
print(regression.coef_)
print(regression.intercept_)
f1 = [[0, 0], [1, 1], [2, 2], [3, 3]]
l1 = regression.predict(f1)
print(l1)
print(regression.score(features, values))
print(regression.score(f1, l1))
print(regression.score(f1, [3, 8, 12, 17]))
"""

import matplotlib.pyplot as plt
from sklearn import linear_model

features = [[0, 1], [1, 3], [2, 6], [3, 7]]  # 已知features點
values = [1, 4, 5.5, 6]  # 已知點數值
regression = linear_model.LinearRegression()
regression.fit(features, values)
print(regression.coef_)
print(regression.intercept_)
f1 = [[0, 0], [1, 1], [2, 2], [3, 3]]  # 新的feature點
l1 = regression.predict(f1)  # 用新的feature點預測數值
print(l1)
estimate_value = regression.predict(features)  # 用已知features點預測數值
print("estimate_value={}".format(estimate_value))
print("real value=", regression.score(features, values))  # 評估準度
print("ideal value=", regression.score(features, estimate_value))
print("idea value=", regression.score(f1, l1))
print("idea value with offset=", regression.score(f1, [3, 8, 12, 17]))
