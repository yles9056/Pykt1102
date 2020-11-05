import numpy as np
from numpy import array
from sklearn.decomposition import PCA

A = array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]])
print(A)
pca = PCA(2)
pca.fit(A)
print('pca.components_\n', pca.components_)
print('pca.explained_variance_\n', pca.explained_variance_)  # 列出重要性
B = pca.transform(A)
print('pca.transform()\n', B)

pca2 = PCA(2)
C = pca2.fit_transform(A)
print(C)
print('pca.components_\n', pca2.components_)
print('pca.explained_variance_\n', pca.explained_variance_)

from joblib import dump, load
dump(pca, "demo46.joblib")
pca3 = load("demo46.joblib")
print('pca.components_\n', pca3.components_)
print('pca.explained_variance_\n', pca3.explained_variance_)
