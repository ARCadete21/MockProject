import pandas as pd
from sklearn.impute import KNNImputer
from .utils import feature_selection_cv


class FeatureSelection:
    def __init__(self, X: pd.DataFrame, y: pd.Series, scaler):
        self.X = X
        self.y = y
        self.scaler = scaler


    def feature_selection(self):
        X_fs = feature_selection_cv(self.X, self.y, self.scaler, KNNImputer())
        return X_fs