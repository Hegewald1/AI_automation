# Breast cancer problem
#
# neural network with keras
# used to make predictions
#
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import optimizers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tensorflow as tf

whole_data = load_breast_cancer()

X_data = whole_data.data
y_data = whole_data.target

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=7)

model = Sequential()

# Keras model with two hidden layer with 10 neurons each
model.add(Dense(32, input_shape=(30,), activation='relu'))  # Input layer => input_shape should be explicitly designated
# model.add(Activation('relu'))
model.add(Dropout(rate=0.1))
model.add(Dense(16, activation='relu'))  # Hidden layer => only output dimension should be designated
model.add(Dropout(rate=0.1))
# model.add(Activation('sigmoid'))
model.add(Dense(8, activation='relu'))                         # Hidden layer => only output dimension should be designated
model.add(Dropout(rate=0.1))
# model.add(Activation('sigmoid'))
model.add(Dense(1, activation='sigmoid'))  # Output layer => output dimension = 1 since it is regression problem
# model.add(Activation('sigmoid'))

'''
This is equivalent to the above code block
>> model.add(Dense(10, input_shape = (13,), activation = 'sigmoid'))
>> model.add(Dense(10, activation = 'sigmoid'))
>> model.add(Dense(10, activation = 'sigmoid'))
>> model.add(Dense(1, activation = 'sigmoid'))
'''

sgd = tf.keras.optimizers.SGD(learning_rate=0.01)  # stochastic gradient descent optimizer
# sgd used as optimizer
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=50, epochs=100, verbose=1)

results = model.evaluate(X_test, y_test)

print('loss: ', results[0])
print('accuracy: ', results[1])
