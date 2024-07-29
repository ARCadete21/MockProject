import pandas as pd


class DataSplit:
    def __init__(self, data: pd.DataFrame, target: str):
        self.data = data
        self.y = None
        self.X = None
        self.target = target
    
    
    def split(self):
        self.y = self.data[self.target]
        self.X = self.data.drop(columns=[self.target])
        return self.X, self.y