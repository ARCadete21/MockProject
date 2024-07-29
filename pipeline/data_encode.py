import pandas as pd
from .utils import one_hot_encoding


class DataEncode:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    
    def onehot_encoding(self):
        data_oh = one_hot_encoding(self.data)
        return data_oh