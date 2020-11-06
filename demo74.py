import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import keras
from keras import callbacks, Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard, Callback


class EarlyStopCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if (logs.get('val_accuracy') > 0.98):
            print("reach 98% accuracy in valudation")
            self.model.stop_training = True


escb1 = EarlyStopCallback()

csv = pd.read_csv('data/demo74_bmi.csv')
print(csv.shape)
print(csv.head(n=10))
csv['height'] = csv['height'] / 200
csv['weight'] = csv['weight'] / 100

encoder = LabelBinarizer()
transformedLabel = encoder.fit_transform(csv['label'])
print(csv['label'][:10])
print(transformedLabel[:10])

DATA_FOR_TEST = 90000
test_csv = csv[DATA_FOR_TEST:]
test_part = test_csv[['weight', 'height']]
test_answer = transformedLabel[DATA_FOR_TEST:]
train_csv = csv[:DATA_FOR_TEST]
train_part = train_csv[['weight', 'height']]
train_answer = transformedLabel[:DATA_FOR_TEST]
print(test_part.shape, train_part.shape, test_answer.shape, train_answer.shape)

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(2,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
board1 = TensorBoard(log_dir="logs/demo74", histogram_freq=1)
history = model.fit(train_part, train_answer, batch_size=100, epochs=200, verbose=1,
                    validation_data=(test_part, test_answer), callbacks=[board1, escb1])
