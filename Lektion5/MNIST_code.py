# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:45:23 2020
@author: Sila
"""


# imports for array-handling and plotting
import numpy as np
import matplotlib.pyplot as plt

# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.metrics import multilabel_confusion_matrix

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(X_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])

plt.show()

# In order to train our neural network to classify images we first have to
# unroll the height width pixel format into one big vector
# - the input vector. 
# But let's graph the distribution of our pixel values.

plt.subplot(2,1,1)
plt.imshow(X_train[0], cmap='gray', interpolation='none')
plt.title("Digit: {}".format(y_train[0]))
plt.xticks([])
plt.yticks([])
plt.subplot(2,1,2)
plt.hist(X_train[0].reshape(784))
plt.title("Pixel Value Distribution")

plt.show()

# The pixel values range from 0 to 255:
# the background majority close to 0, and those close to 255 representing the digit.

#Let's reshape our inputs to a single vector vector
# and normalize the pixel values to lie between 0 and 1.

# let's print the shape before we reshape and normalize
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)

# What values do we have
print("Our values in the training set")
print(np.unique(y_train, return_counts=True))

#Let's encode our categories - digits from 0 to 9 - using one-hot encoding.
# The result is a vector with a length equal to the number of categories.
# The vector is all zeroes except in the position for the respective category.
# Thus a '4' will be represented by [0,0,0,1,0,0,0,0,0].

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# building a linear stack of layers with the sequential model
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

# compiling the sequential model
#Here we specify our loss function (or objective function).
# We use cross entropy, other loss functions will probably work as well.
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

#Having compiled our model we can now start the training process.
# We have to specify how many times we want to iterate on the whole training set (epochs)
# and how many samples we use for one update to the model's weights (batch size).

# training the model and saving metrics in history
history = model.fit(X_train, Y_train,
          batch_size=128, epochs=2,
          verbose=2,
          validation_data=(X_test, Y_test))

# plotting the metrics
#fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

plt.show()

loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])

# predicted_classes = model.predict_classes(X_test)
predicted_classes = np.argmax(model.predict(X_test), axis=-1)
#predicted_classes = model.predict_step(X_test)
#print(multilabel_confusion_matrix(Y_test, predicted_classes))

# see which we predicted correctly and which not
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print()
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")

# adapt figure size to accomodate 18 subplots
plt.rcParams['figure.figsize'] = (7,14)

# plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted: {}, Truth: {}".format(predicted_classes[correct],
                                        y_test[correct]))
    plt.xticks([])
    plt.yticks([])

plt.show()
# plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6,3,i+10)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted {}, Truth: {}".format(predicted_classes[incorrect],
                                       y_test[incorrect]))
    plt.xticks([])
    plt.yticks([])

plt.show()
