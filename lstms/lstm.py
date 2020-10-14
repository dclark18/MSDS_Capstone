import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras import layers
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocess import MinMaxScaler
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class LSTMPipeline:

    def __init__(self, data_path: str, bear_id: int):

        self.raw_data = pd.read_csv(data_path)
        self.bear_id = bear_id

    def preprocess_stage(self) -> Tuple[MinMaxScaler, np.array]:
        """
        Read in data, set the datetime column, and do the test/train split.

        Returns:
        Train/test split dataset
        """

        df_formatted = self.raw_data.copy()
        df_formatted = df_formatted.loc[df_formatted.bear_id == self.bear_id,
                                        ['distrdsMIN', 'datetime']]
        df_formatted.datetime = pd.to_datetime(df_formatted.datetime)
        df_formatted = df_formatted.set_index('datetime')

        # Normalize
        m = MinMaxScaler()
        df_scaled = m.fit_transform(df_formatted)

        return m, df_scaled

    def fit_model_stage(self, scaled_vals: np.array):

        # Complete prototype: split naively.
        split_len = int(0.7 * len(scaled_vals))

        train, test = scaled_vals[:split_len], scaled_vals[split_len:]

        # Example taken from
        # https://medium.com/@cdabakoglu/time-series-forecasting-arima-lstm-prophet-with-python-e73a750a9887

        n_input = len(train)
        n_features = 1
        generator = TimeseriesGenerator(train,
                                        train,
                                        length=n_input,
                                        batch_size=1)

        lstm_model = keras.models.Sequential()
        lstm_model.add(layers.LSTM(4, activation='relu',
                                   input_shape=(n_input, n_features)))
        lstm_model.add(layers.Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')

        # Fit the model
        lstm_model.fit_generator(generator, epochs=20)
        return lstm_model, test

    def predict_model_stage(self, model, test_data):

        outputs = model.predict(test_data)
        return outputs

    def main(self):

        transform, data_scaled = self.preprocess_stage()

        model, test_data = self.fit_model_stage(data_scaled)

        outputs = self.predict_model_stage(model, test_data)
        breakpoint()

        return outputs


if __name__ == '__main__':

    pipeline = LSTMPipeline('/users/shawd/Capstone/Archive/femaleclean4.csv')
