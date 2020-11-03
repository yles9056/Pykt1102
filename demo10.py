from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from matplotlib import pyplot

X, y = make_regression(n_samples=1000, n_features=10, n_informative=5)
"""
X: features
y: labels
n_informative: 真正有用的feature個數
"""
model = LinearRegression()
model.fit(X, y)
importance = model.coef_
for index, value in enumerate(importance):
    print("#{} feature score={:.3f} ".format(index, value))
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

kBest = SelectKBest(f_regression, k=4).fit(X, y)  # k: 取係數較大的前k個features
print(kBest.get_support())  # 列出那些features被選中
newX = kBest.fit_transform(X, y)  # 從原來的features中挑出新的features
print(X[0])
print(newX[0])
