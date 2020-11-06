

import numpy as np
from keras import layers, models
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# Words are ranked by how often they occur (in the training set) and only the num_words most frequent words are kept.
print(train_data[0])
print(train_data[1])
print(max([max(sequence) for sequence in train_data]))
word_index = imdb.get_word_index()
reverse_word_index = dict((v, k) for k, v in word_index.items())
for j in range(5):
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[j]])
    print(decoded_review)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        results[i, seq] = 1.  # 影評中有出現這個詞則標記為1
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.array(train_labels).astype('float32')
y_test = np.array(test_labels).astype('float32')
print(x_train[0])

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(10000,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss="binary_crossentropy",
              metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, epochs=20, batch_size=100)
