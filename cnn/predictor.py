import os
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
from data_processor import DataProcessor
from sklearn.metrics import accuracy_score


class Predictor:
    def __init__(self) -> None:
        self.current_dir = os.path.dirname(__file__)

    def call(self):
        trained_model = self.load_trained_model("cnn.keras")
        _, X_test_reshaped, _, y_test = self.load_inputs()
        custom_inputs = self.load_custom_inputs()

        self.predict(trained_model, X_test_reshaped, y_test)
        self.predict_custom_unputs(trained_model, custom_inputs)

    def load_trained_model(self, model_name):
        print('Loading trained model')
        model_path = os.path.join(self.current_dir, "model", model_name)
        return load_model(model_path)

    def load_inputs(self):
        print('Loading inputs')
        return DataProcessor().preprocess_data()

    def load_custom_inputs(self):
        print('Loading custom inputs')
        parent_dir = os.path.dirname(self.current_dir)
        input_dir = os.path.join(parent_dir, "dataset", "my_inputs")
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
            image_arrays.append(reshaped_image)

        return np.array(image_arrays)

    def predict(self, model, X_test_reshaped, y_test):
        print('Predict inputs')
        predictions = model.predict(X_test_reshaped)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(y_test, predicted_classes)
        print("Accuracy:", accuracy)
        test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test)
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)

    def predict_custom_unputs(self, trained_model, inputs):
        print('Predict custom inputs')
        i = -1
        for image in inputs:
            i += 1
            prediction = trained_model.predict(image)
            print(f"Predicted image with digit {i}:", np.argmax(prediction))


Predictor().call()
