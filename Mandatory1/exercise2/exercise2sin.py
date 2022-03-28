import math

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten

# select 2500 random floats from interval 0 to 10
x = np.random.uniform(low=0, high=10, size=(2500, 1))
y = np.sin(x)

model = Sequential()
model.add(Dense(100, input_dim=1, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=150, batch_size=50)
model.summary()
# predictions = model.predict([-1.5, 0.8, 4.2, 5])
# print(predictions)  # Approximately -, -, -, -

y_new = model.predict(x)

plt.subplot(2, 1, 1)
plt.scatter(x, y, color='blue', s=1)
plt.suptitle('y = sin(x)')
plt.ylabel('Real y')
plt.grid()

plt.subplot(2, 1, 2)
plt.scatter(x, y_new, color='red', s=1)
plt.xlabel('x')
plt.ylabel('Approximated y')
plt.grid()

plt.show()
