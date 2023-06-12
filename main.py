# importing required libraries

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.activations import softmax


# downloading mnist dataset

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # data loading

plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()  # plotting the first image of training data
#print(x_train[0])

# normalizing data

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
#print(x_train[0])

# building model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training
model.fit(x_train, y_train, epochs=12)

# loss and accuracy

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save('mnist_tf.model')
