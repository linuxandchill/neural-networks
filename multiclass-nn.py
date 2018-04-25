from keras import layers
from keras import models
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
import numpy as np

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
        num_words = 10000)

def vectorize_seq(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results 

#vectorize data
x_train = vectorize_seq(train_data)
x_test = vectorize_seq(test_data)
#vectorize labels with one-hot encoding 
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

model = models.Sequential()
#avoid hidden/intermediate layers with fewer units than output
#don't create bottleneck
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
#use softmax to output probability distribution
#scores will sum to 1
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train= x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=10,
                    batch_size=512,
                    validation_data=(x_val, y_val))

import matplotlib.pyplot as plt
train_loss = history.history['loss']
val_loss= history.history['val_loss']
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'bo', label="training loss")
plt.plot(epochs, val_loss, 'b', label="Validation loss")
plt.title("Validation and training loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()
train_acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, train_acc, 'bo', label="training acc")
plt.plot(epochs, val_acc, 'b', label="validation acc")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()  

results = model.evaluate(x_test, y_test)

#probability distribution?
predictions = model.predict(x_test)
#predictions should add to 1
np.sum(predictions[0])

