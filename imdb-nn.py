from keras import layers
from keras import models
from keras.datasets import imdb
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words = 10000)

#decode train_data back to words (for fun)
word_index = imdb.get_word_index()
reverse_word_index = dict(
        [(value,key) for (key,value) in word_index.items()])
decoded_review = ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in train_data[0]])

def vectorize_seq(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results 

#vectorize data
x_train = vectorize_seq(train_data)
x_test = vectorize_seq(test_data)
#vectorize labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#build model
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))
#model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


#validation to monitor during training
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#train
model.compile(optimizer='rmsprop', 
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 10,
                    batch_size = 512,
                    validation_data=(x_val, y_val))

#history
history_dict = history.history
history_dict.keys()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['acc']
#plot metrics for loss
epochs = range(1, len(acc)+1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values,'b', label='Validation loss')
plt.title('Training and val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
#plot metrics for acccuracy
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
acc = history_dict['acc']
epochs = range(1, len(acc)+1)
plt.plot(epochs, acc_values, 'bo', label= 'Training acc')
plt.plot(epochs,val_acc_values, 'b', label='Validation acc')
plt.title('Training and validaiton accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#see how model performs on test set
results = model.evaluate(x_test,y_test)

#confidence on samples
model.predict(x_test)




