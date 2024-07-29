import pickle
import pandas as pd


class DataColumns:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.columns = None


    def _get_columns(self):
        self.columns = list(self.data.columns)


    def dump_columns(self):
        self._get_columns()
        pickle.dump(self.columns, open('pipeline/support/aposteriori/columns.pkl', 'wb'))