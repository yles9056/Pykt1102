from numpy import array, cov, mean
from numpy.linalg import eig

A = array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]])
print('A\n', A)
M = mean(A.T, axis=1)
print('M\n', M)
M2 = mean(A.T)
print('M2\n', M2)
M3 = mean(A, axis=1)
print('M3\n', M3)
C = A - M
print('C\n', C)
V = cov(C.T)
print('V\n', V)
values, vectors = eig(V)
print("values=", values)
print("vectors=", vectors)
P = vectors.T.dot(C.T)
print('P\n', P.T)
