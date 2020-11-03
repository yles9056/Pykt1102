# ndarray.view(), ndarray.copy(), reference

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = a  # 參照
c = a.view()  # 資料參照，但有自己的shape
d = a.copy()  # 複製一份資料成為獨立資料
print('a', a)
print('b', b)
print('c', c)
print('d', d)
print('----')
a[0][0] = 100
print('a', a)
print('b', b)
print('c', c)
print('d', d)
print('----')
a.shape = (4,)
print('a', a)
print('b', b)
print('c', c)
print('d', d)
print('----')
