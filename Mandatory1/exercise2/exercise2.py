import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten

# select 5000 random floats from interval -2 to 6
x = np.random.uniform(low=-2, high=6, size=(5000, 1))
y = x*x*x - 6*x**2 + 4*x + 12

model = Sequential()
model.add(Dense(80, input_dim=1, activation='sigmoid'))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=150, batch_size=50)


predictions = model.predict([-1.5, 0.8, 4.2, 5])
print(predictions)  # Approximately -10.875, 11.872, -2.952, 7

y_new = model.predict(x)

plt.subplot(2, 1, 1)
plt.scatter(x, y, color='blue', s=1)
plt.suptitle('y = $x^3$ - 6$x^2$ + 4x + 12')
plt.ylabel('Real y')
plt.grid()

plt.subplot(2, 1, 2)
plt.scatter(x, y_new, color='red', s=1)
plt.xlabel('x')
plt.ylabel('Approximated y')
plt.grid()

plt.show()
