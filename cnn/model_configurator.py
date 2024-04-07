from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input


class ModelConfigurator:
    def create_compiled_model(self):
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

        model.compile(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']
        )

        return model
