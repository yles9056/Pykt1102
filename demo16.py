# joining ndarray

import numpy as np

a1 = np.array([1, 2])
a2 = np.array([3, 4])
a3 = np.array([[1, 2], [3, 4]])
a4 = np.array([[5, 6], [7, 8]])
"""
c_: column
r_: row
"""
print('np.c_[a1, a2]')
print(np.c_[a1, a2])
print('np.c_[a1, a2, a1]')
print(np.c_[a1, a2, a1])
print('np.c_[a3, a4]')
print(np.c_[a3, a4])
print('np.r_[a1, a2]')
print(np.r_[a1, a2])
print('np.r_[a1, a2, a1]')
print(np.r_[a1, a2, a1])
print('np.r_[a3, a4]')
print(np.r_[a3, a4])
print('np.hstack((a1, a2))')
print(np.hstack((a1, a2)))
print('np.vstack((a1, a2))')
print(np.vstack((a1, a2)))
print('np.hstack((a3, a4))')
print(np.hstack((a3, a4)))
print('np.vstack((a3, a4))')
print(np.vstack((a3, a4)))
