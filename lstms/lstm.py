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

    def __init__(self, data_path: str, bear_id: int, window_size: int):
        """
        Args:
            data_path: filepath of the bear data to read in
            bear_id: select a particular bear
            window_size: length of the time series to model. 
                ex. Based on the last n time steps, predict the next value
        """

        self.raw_data = pd.read_csv(data_path)
        self.bear_id = bear_id
        self.window_size = window_size

        if bear_id not in self.raw_data.Bear_ID:
            raise ValueError(f"Specified bear ID {self.bear_id} "
                             f"not in {self.raw_data.bear_id.unique()}")

        # Create a scaler object since we need it to back-transform outputs
        self.scaler = MinMaxScaler()

    def preprocess_stage(self) -> Tuple[np.array, np.array]:
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

        # Store the formatted dataset for later
        self.data_formatted = df_formatted

        # Complete prototype: split naively.
        split_len = int(0.7 * len(df_formatted))

        train, test = df_formatted[:split_len], df_formatted[split_len:]

        # Normalize
        self.scaler.fit(train)
        train_scaled = self.scaler.transform(train)
        test_scaled = self.scaler.transform(test)

        return train_scaled, test_scaled

    def fit_model_stage(self, train: np.array) -> keras.models.Sequential:

        # Example taken from
        # https://medium.com/@cdabakoglu/time-series-forecasting-arima-lstm-prophet-with-python-e73a750a9887

        n_features = 1
        batch_size = 64
        length = self.window_size
        generator = TimeseriesGenerator(train, train,
                                        length=length, batch_size=batch_size)
        
        lstm_model = keras.models.Sequential()
        lstm_model.add(layers.LSTM(40, activation='relu',
                                   input_shape=(None, n_features),
                                   # batch_size=batch_size,
                                   stateful=False))  # Stateful true for correlated long-term predictions
        lstm_model.add(layers.Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')

        # Fit the model
        lstm_model.fit(generator, epochs=20)
        return lstm_model

    def predict_model_stage(self, model, dataset, start_idx):
        """ Predict out using a moving time window

        Args:
        model - the fit LSTM
        dataset - full dataset, train and test. used to feed inputs for prediction
        start_idx- where the train/test divide index is. We want to validate against the test dataset
        """

        # Logic: for a call with window=5, and starting index of 500:
        # we want to use the last five datapoints to predict the value at index 500
        # then compare to index 500 at the end

        outputs = []

        for window_end in range(start_idx, len(dataset)):

            window_start = window_end - self.window_size
            pred_input = dataset[window_start:window_end]
            pred_input_reshaped = pred_input.reshape((1, window, 1))
            output = model.predict(pred_input_reshaped)
            outputs.append(output)

        # Reverse the minmax transformation
        outputs_stacked = np.stack(outputs).reshape(len(outputs), 1)
        outputs_unscaled = self.scaler.inverse_transform(outputs_stacked)
        test_unscaled = self.scaler.inverse_transform(dataset)

        # Plot
        with PdfPages('bear_plot.pdf') as pdf:
            plt.plot(outputs_unscaled, 'r-', test_unscaled[start_idx:], 'b-')
            plt.title(f"Plot with window of {self.window_size}")
            pdf.savefig()
            plt.close()


    def main(self):

        # Preprocessing. Train/test split, and min-max scale
        train_data, test_data = self.preprocess_stage()

        # Fit the LSTM
        model = self.fit_model_stage(train_data)

        # Create predictions and plot. 
        full_data = np.concatenate([train_data, test_data])
        start_idx = len(train_data)
        outputs = self.predict_model_stage(model, full_data, start_idx, 5)



if __name__ == '__main__':
    bear_path = 'femaleclean4.csv'
    pipeline = LSTMPipeline(bear_path, 7)

    outputs = pipeline.main()
