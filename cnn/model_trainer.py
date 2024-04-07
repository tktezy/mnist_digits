import os
from model_configurator import ModelConfigurator
from data_processor import DataProcessor
from keras.callbacks import EarlyStopping


class ModelTrainer:
    def train(self):
        print('Loading model')
        model = self.create_compiled_model()

        print('Preprocessing data')
        X_train_reshaped, X_test_reshaped, y_train, y_test = self.preprocess_data()

        print('Training model')
        model.fit(
            X_train_reshaped,
            y_train,
            epochs=100,
            batch_size=128,
            validation_data=(X_test_reshaped, y_test),
            callbacks=[self.early_stopping()]
        )

        model.summary()
        print('Saving model')
        model.save(self.model_path())

    def create_compiled_model(self):
        return ModelConfigurator().create_compiled_model()

    def preprocess_data(self):
        return DataProcessor().preprocess_data()

    def early_stopping(self):
        return EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    def model_path(self):
        current_dir = os.path.dirname(__file__)
        return os.path.join(current_dir, "model", "cnn.keras")


ModelTrainer().train()
