import pandas as pd


class LoadData:
    def __init__(self, data_path):
        self.data_path = data_path


    def load_data(self):
        self.data = pd.read_csv(self.data_path, index_col=0)
        return self.data