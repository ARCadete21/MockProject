from pipeline.load_data import LoadData
from pipeline.preprocess_data import Preprocessor
from pipeline.data_encode import DataEncode
from pipeline.data_split import DataSplit
from pipeline.feature_selection import FeatureSelection
from pipeline.get_columns import DataColumns
from pipeline.train_model import TrainModel
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier


def main(model, scaler):
    data = LoadData('data/train.csv').load_data()
    data = Preprocessor(data).preprocessor()
    X, y = DataSplit(data, 'Success').split()
    X = DataEncode(X).onehot_encoding()
    X = FeatureSelection(X, y, scaler).feature_selection()
    DataColumns(X).dump_columns()
    TrainModel(X, y, model, scaler).dump_pipeline()


mlp = MLPClassifier(
    activation='tanh', alpha=0.005, hidden_layer_sizes=10, batch_size=256,
    learning_rate='adaptive', solver='sgd', learning_rate_init=0.01, random_state=16
    )

scaler = RobustScaler()


if __name__ == '__main__':
    main(mlp, scaler)