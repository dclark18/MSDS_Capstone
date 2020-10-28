import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras import layers
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class LSTMPipeline:

    def __init__(self, data_path: str, bear_id: int):

        self.raw_data = pd.read_csv(data_path)
        self.bear_id = bear_id

        if bear_id not in self.raw_data.Bear_ID:
            raise ValueError(f"Specified bear ID {self.bear_id} "
                             f"not in {self.raw_data.bear_id.unique()}")

    def preprocess_stage(self) -> Tuple[MinMaxScaler, np.array, np.array]:
        """
        Read in data, set the datetime column, and do the test/train split.

        Returns:
        Train/test split dataset
        """

        df_formatted = self.raw_data.copy()
        df_formatted = df_formatted.loc[df_formatted.Bear_ID == self.bear_id,
                                        ['distrdsMIN', 'datetime']]
        df_formatted.datetime = pd.to_datetime(df_formatted.datetime)
        df_formatted = df_formatted.set_index('datetime')

        # Complete prototype: split naively.
        split_len = int(0.7 * len(df_formatted))

        train, test = df_formatted[:split_len], df_formatted[split_len:]

        # Normalize
        m = MinMaxScaler()
        train_scaled = m.fit_transform(train)
        test_scaled = m.transform(test)

        return m, train_scaled, test_scaled

    def fit_model_stage(self, train: np.array) -> keras.models.Sequential:

        # Example taken from
        # https://medium.com/@cdabakoglu/time-series-forecasting-arima-lstm-prophet-with-python-e73a750a9887

        n_input = len(train) - 1
        n_features = 1
        generator = TimeseriesGenerator(train, train,
                                        length=n_input, batch_size=1)

        lstm_model = keras.models.Sequential()
        lstm_model.add(layers.LSTM(4, activation='relu',
                                   input_shape=(n_input, n_features),
                                   batch_size=1,
                                   stateful=True))
        lstm_model.add(layers.Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')

        # Fit the model
        lstm_model.fit(generator, epochs=20)
        return lstm_model

    def predict_model_stage(self, model, test_data) -> np.array:

        outputs = model.predict(test_data)
        return outputs

    def main(self):

        transform, train_data, test_data = self.preprocess_stage()

        model = self.fit_model_stage(train_data)
        outputs = self.predict_model_stage(model, test_data)

        return outputs


if __name__ == '__main__':
    bear_path = '/users/shawd/Capstone/Archive/femaleclean4.csv'
    pipeline = LSTMPipeline(bear_path, 7)

    outputs = pipeline.main()
