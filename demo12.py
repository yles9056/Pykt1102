# ndarray.view(), np.reshape()

import numpy as np

a = np.zeros((10, 2))
print('a', a)
b = a.T
print('b', b)
c = b.view()
print('c', c)
d = np.reshape(b, (5, 4))
e = np.reshape(b, (20,))
f = np.reshape(b, (20, 1))
g = np.reshape(b, (-1, 20))
print('d.shape', d.shape)
print('e.shape', e.shape)
print('f.shape', f.shape)
print('g.shape', g.shape)
print('d', d)
print('e', e)
print('f', f)
print('g', g)
