# import tensorflow as tf

# print("TensorFlow version:", tf.__version__)

# tf.config.set_visible_devices([], 'GPU')
# cifar = tf.keras.datasets.cifar100
# (x_train, y_train), (x_test, y_test) = cifar.load_data()
# model = tf.keras.applications.ResNet50(
#     include_top=True,
#     weights=None,
#     input_shape=(32, 32, 3),
#     classes=100,)

# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=5, batch_size=64)

# import numpy as np
# import tensorflow as tf
# print(tf.__version__)
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras import models
# from tensorflow.keras import layers

# tf.config.set_visible_devicews([], 'GPU')

# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# train_images = train_images.reshape((60000, 28*28))
# train_images = train_images.astype("float32") / 255

# test_images = test_images.reshape((10000,28*28,1))
# test_images = test_images.astype("float32") /255

# # Convert the data labels to categorical values by performing one-hot encoding
# train_labels = tf.keras.utils.to_categorical(train_labels)
# test_labels = tf.keras.utils.to_categorical(test_labels)

# network = models.Sequential()
# network.add(layers.Dense(512, activation="relu", input_shape=(28*28,)))
# network.add(layers.Dense(10, activation="softmax"))

# network.compile(optimizer="adam",loss="categorical_crossentropy" ,metrics=["accuracy"])
# network.fit(train_images, train_labels, epochs=5, batch_size=128)
