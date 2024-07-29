import pickle
import numpy as np


class ModelPredictor:
    def __init__(self):
        self.columns = self._load_columns('pipeline/support/aposteriori/columns.pkl')
        self.pipeline = self._load_pipeline('pipeline/support/aposteriori/pipeline.pkl')


    @staticmethod
    def _load_columns(filepath):
        try:
            with open(filepath, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            raise Exception(f"Column file not found at {filepath}")
    

    @staticmethod
    def _load_pipeline(filepath):
        try:
            with open(filepath, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            raise Exception(f"Pipeline file not found at {filepath}")


    def _prepare_input_data(self, data):
        inputs_dict = {col: data.get(col, None) for col in self.columns}
        if None in inputs_dict.values():
            raise ValueError("Missing data for one or more columns")

        try:
            test_arr = np.array([float(inputs_dict[col]) for col in self.columns]).reshape(1, -1)
            return test_arr
        except ValueError as e:
            raise ValueError(f"Error converting input values: {e}")


    def get_prediction(self, data):
        test_arr = self._prepare_input_data(data)
        prediction = self.pipeline.predict(test_arr)
        return f"The student is likely to {str(prediction[0]).lower()}"