import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

I = 0
plt.imshow(x_test[I].reshape([28, 28]), cmap='gray')


model.fit(x_train, y_train_cat, epochs=5)
model.evaluate(x_test, y_test_cat)

plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i], cmap=plt.cm.binary)
plt.show()

print("Мнение нейронной сети: ", np.argmax(model.predict(x_test[I].reshape([1, 28, 28]))))
print("Верный ответ: ", y_test[I])

for i in range(I+2):
    print(y_test[i])