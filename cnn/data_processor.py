import os
import pandas as pd


class DataProcessor:
    def preprocess_data(self):
        train_path = self.load_data("mnist_train.csv")
        test_path = self.load_data("mnist_test.csv")

        X_train = self.clean_X_data(train_path)
        X_test = self.clean_X_data(test_path)
        y_train = self.clean_y_data(train_path)
        y_test = self.clean_y_data(test_path)

        X_train_reshaped = self.reshape_data(X_train)
        X_test_reshaped = self.reshape_data(X_test)

        return X_train_reshaped, X_test_reshaped, y_train, y_test

    def load_data(self, filename):
        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(current_dir)
        return os.path.join(parent_dir, "dataset", filename)

    def clean_X_data(self, data_path):
        return pd.read_csv(data_path, header=None).drop(columns=[0]).values

    def clean_y_data(self, data_path):
        return pd.read_csv(data_path, header=None)[0].astype(int)

    def reshape_data(self, data):
        return data.reshape(-1, 28, 28, 1)
