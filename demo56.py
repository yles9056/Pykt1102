# first deep learning model

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback


class EarlyStopCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('loss') < 0.5:
            print("\n**loss < 0.5, can stop**\n")
            self.model.stop_training = True


callback1 = EarlyStopCallback()

DATA_FILE = 'data/diabetes.csv'
dataset1 = np.loadtxt(DATA_FILE, delimiter=',', skiprows=1)
print(dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]

model = Sequential()
model.add(Dense(14, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
"""
x: (8+1)*14=126
Y: (14+1)*8=120
Z: (8+1)*8=9
"""
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(inputList, resultList, epochs=200, batch_size=20, callbacks=[callback1])
score = model.evaluate(inputList, resultList)
print("score=", score)
for s, n in zip(score, model.metrics_names):
    print("{} value={}".format(n, s))
