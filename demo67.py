# softmax

import numpy as np
import tensorflow as tf

scores = [3, 1, 2]

def ratio(x):
    y = np.array(x)
    return y / np.sum(y, axis=0)


def manualSoftMax(x):
    y = np.array(x)
    return np.exp(y) / np.sum(np.exp(y), axis=0)


print(ratio(scores))
print(manualSoftMax(scores))
print(tf.nn.softmax(np.array(scores, dtype=float)).numpy())
