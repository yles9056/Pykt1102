import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

regressionData = datasets.make_regression(100, 1, noise=15)
"""
arg1: feature數
arg2: label數
noise: 雜訊
"""
print(type(regressionData))
features = regressionData[0]
labels = regressionData[1]
# Debug時用View as Array觀看矩陣數值
print(type(features), features.shape)
print(type(labels), labels.shape)
plt.scatter(features, labels, c='red', marker='.')
plt.show()
regression1 = linear_model.LinearRegression()
regression1.fit(regressionData[0], regressionData[1])
print("coef={}, intercept={}".format(regression1.coef_[0], regression1.intercept_))
range1 = [-3, 3]
plt.plot(range1, regression1.coef_ * range1 + regression1.intercept_, c='blue')
plt.scatter(features, labels, c='red', marker='.')
plt.show()
print(regression1.score(features, labels))
