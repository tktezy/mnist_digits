import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from PIL import Image, ImageOps
from keras.callbacks import EarlyStopping


# tf.config.set_visible_devices([], 'GPU')

print('Loading data')
current_dir = os.path.dirname(__file__)
train_image_path = os.path.join(current_dir, "dataset", "mnist_train.csv")
test_image_path = os.path.join(current_dir, "dataset", "mnist_test.csv")
X_train = pd.read_csv(train_image_path, header=None).drop(columns=[0]).values
y_train = pd.read_csv(train_image_path, header=None)[0]
X_train, X_val, y_train, y_val = train_test_split(X_train_reshaped, y_train, test_size=0.2, random_state=42)

X_test = pd.read_csv(test_image_path, header=None).drop(columns=[0]).values
y_test = pd.read_csv(test_image_path, header=None)[0]

print('Reshaping data')
X_train_reshaped = X_train.reshape(-1, 28, 28, 1)
X_test_reshaped = X_test.reshape(-1, 28, 28, 1)

print('Defining and compiling model')
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    Conv2D(256, kernel_size=(3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

model_path = os.path.join(current_dir, "model", "cnn.keras")
if os.path.exists(model_path):
    print('Loading model')
    model = load_model(model_path)
else:
    print('Training model')
    model.fit(
        X_train_reshaped,
        y_train,
        epochs=100,
        batch_size=128,
        validation_data=(X_test_reshaped, y_test),
        callbacks=[early_stopping]
        )
    print('Saving model')
    model.save(model_path)

print('================================================================================================')
print(model.summary())
print('================================================================================================')
test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print('================================================================================================')

print('Loading inputs')
input_dir = os.path.join(current_dir, "dataset", "my_inputs")
image_paths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]
sorted_image_paths = sorted(image_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

image_arrays = []
for image_path in sorted_image_paths:
    image = Image.open(image_path)
    resized_image = image.resize((28, 28))
    gray_image = resized_image.convert("L")
    inverted_image = ImageOps.invert(gray_image)

    image_array = np.array(inverted_image)
    reshaped_image = image_array.reshape(-1, 28, 28, 1)
    # plt.imshow(reshaped_image[0], cmap='gray')
    # plt.title('Image')
    # plt.axis('off')
    # plt.show()
    image_arrays.append(reshaped_image)

image_data = np.array(image_arrays)

i = -1
for image in image_data:
    i += 1
    prediction = model.predict(image)
    print(f"Predicted image with digit {i}:", np.argmax(prediction))
