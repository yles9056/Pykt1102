# n_clusters and inertia_

# 測試不同的k, n_clusters，找到轉折點
# inertia低一點比

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

X = np.r_[np.random.randn(10000, 2) + [2, 2],
          np.random.randn(10000, 2) + [0, -2],
          np.random.randn(10000, 2) + [-2, 2]]
print(X.shape)
interias = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    interias.append(kmeans.inertia_)
print(interias)
plt.plot(range(1, 10), interias)
plt.show()
