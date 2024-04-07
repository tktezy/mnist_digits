import os
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

tf.config.set_visible_devices([], 'GPU')

print('Loadnig data')
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
train_data_path = os.path.join(parent_dir, "dataset", "mnist_train.csv")
test_data_path = os.path.join(parent_dir, "dataset", "mnist_test.csv")
X_train = pd.read_csv(train_data_path, header=None).drop(columns=[0]).values.astype('float32') / 255
y_train = pd.read_csv(train_data_path, header=None)[0].values.ravel()
X_test = pd.read_csv(test_data_path, header=None).drop(columns=[0]).values.astype('float32') / 255
y_test = pd.read_csv(test_data_path, header=None)[0].values.ravel()

print('Converting labels')
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
num_classes = len(label_encoder.classes_)
y_train_onehot = to_categorical(y_train_encoded, num_classes=num_classes)
y_test_onehot = to_categorical(y_test_encoded, num_classes=num_classes)

print('Defining model')
model = Sequential([
    Input(shape=[784]),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

print('Compiling model')
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
)

model_path = os.path.join(current_dir, "model", "mlp.keras")
if os.path.exists(model_path):
    print('Loading model')
    model = load_model(model_path)
else:
    print('Training model')
    model.fit(
        X_train, y_train_onehot, epochs=10, batch_size=64, validation_data=(X_test, y_test_onehot)
    )
    print('Saving model')
    model.save(model_path)

print('Summary', model.summary())
loss, accuracy = model.evaluate(X_test, y_test_onehot)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
