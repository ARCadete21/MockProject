import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer


class TrainModel:
    def __init__(self, X: pd.DataFrame, y: pd.Series, model, scaler):
        self.X = X
        self.y = y
        self.model = model
        self.scaler = scaler
        self.pipeline = None


    def _define_pipeline(self):
        self.pipeline = Pipeline(steps=[
            ('scaler', self.scaler),
            ('imputer', KNNImputer()),
            ('classifier', self.model)
        ])


    def _train(self):
        self._define_pipeline()
        self.pipeline.fit(self.X, self.y)


    def dump_pipeline(self):
        self._train()
        pickle.dump(self.pipeline, open('pipeline/support/aposteriori/pipeline.pkl', 'wb'))