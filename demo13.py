# ndarray.view(), reference

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = a.view()  # 資料參照，但有自己的shape
print('a', a)
print('b', b)
print('---')
a.shape = (4, -1)
print('a', a)
print('b', b)
c = a
c.shape = (1, 4)
print('---')
print('a', a)
print('b', b)
print('c', c)
