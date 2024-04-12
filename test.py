import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

cloth = {
    0: "Футболка / топ",
    1: "Штаны",
    2: "Свитер",
    3: "Платье",
    4: "Пальто",
    5: "Сандали",
    6: "Рубашка",
    7: "Кроссовок",
    8: "Сумка",
    9: "Ботинок"
}

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
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

print("Мнение нейронной сети: ", cloth[np.argmax(model.predict(x_test[I].reshape([1, 28, 28])))])
print("Верный ответ: ", cloth[y_test[I]])

for i in range(I+2):
    print(cloth[y_test[i]])