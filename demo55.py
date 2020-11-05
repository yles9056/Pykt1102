import tensorflow as tf
import keras
import numpy as np


def steak_model(x_new):  # 預測牛排盎司VS價錢
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)  # 盎司
    ys = np.array([500, 1000, 1500, 2000, 2500, 3000], dtype=float)  # 價錢
    layers = [keras.layers.Dense(units=1, input_shape=[1])]
    model = keras.Sequential(layers)
    model.compile(optimizer='sgd', loss="mean_squared_error")
    model.fit(xs, ys, epochs=5000)
    print(model.summary())
    return model.predict(x_new)[0]


print(steak_model([6]))

"""
pycharm/terminal
tensorboard --logdir logs\func20201105-115509
http://localhost:6006/#graphs&run=.
"""
