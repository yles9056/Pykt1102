# NearestNeighbors

import numpy as np
from sklearn.neighbors import NearestNeighbors

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# n_neighbors = 2==>3
shortestNeighbor = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X)
# 找n_neighbors個最近點，自己也算是一個點
distances, indices = shortestNeighbor.kneighbors(X, return_distance=True)
print(distances)
print(indices)
print(shortestNeighbor.kneighbors_graph(X).toarray())