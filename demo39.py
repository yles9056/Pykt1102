# naive_bayes.GaussianNB, partial_fit

import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2],
              [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB()
clf.fit(X, Y)
print(clf.predict([[-0.8, -1], [0.5, 0.5], [-0.5, 0.5], [0, 0]]))

clf_pf = GaussianNB()
# partial_fit: 只fit一部分數據，可以再新增數據點
clf_pf.partial_fit(X, Y, np.unique(Y))  # x, y, classes
# print(clf_pf.predict([[-0.8, -1]]))
print(clf_pf.predict([[0, 0]]))
clf_pf.partial_fit([[0.5, 0.5]], [2])
print(clf_pf.predict([[0, 0]]))
# clf_pf.partial_fit([[0, 0]], [1])
# clf_pf.partial_fit([[-0.7, -0.9]], [2])
# print(clf_pf.predict([[-0.8, -1]]))

