import pandas as pd
import numpy as np

import os
from pathlib import Path
from loguru import logger
from typing import Tuple

from tensorflow import keras

from TimeSeriesPipeline import TimeSeriesPipeline
from utils import unpack_data, plot_model


class MultiVarLSTM(TimeSeriesPipeline):

    def __init__(self, raw_data: pd.DataFrame, window_size: int):

        super().__init__(raw_data, window_size)

    def preprocess(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create the train set and the wandering variable
        """
        logger.info("Starting preprocessing")
        return super().preprocess()

    def fit_model(self,
                  x_data: pd.DataFrame,
                  y_data: pd.Series) -> Tuple[np.ndarray]:
        """
        Fit the multivariate LSTM, and generate predictions.

        Args:
           x_data, y_data: the results from preprocessing

        Output: arrays representing predicted and observed classifications
        """
        logger.info("Fitting the model")

        lstm_model = keras.models.Sequential()
        lstm_model.add(keras.layers.LSTM(40, stateful=False))
        # lstm_model.add(keras.layers.Attention())
        lstm_model.add(keras.layers.Dense(1))

        # Define arguments for model.compile call
        optimizer_kwargs = {'optimizer': 'adam', 'loss': 'mse'}

        # Fit the model. Iterate by bear ID
        predicted, observed = super().loop_and_fit(
            x_data, y_data, lstm_model, optimizer_kwargs)
        return predicted, observed

    def evaluate_model(self, predicted: np.array, observed: np.array) -> None:
        """
        Given labels, return the accuracy and a ROC curve.
        """

        output_path = Path(os.path.abspath(__file__)).parent
        df_path = output_path / 'lstm_predictions.csv'
        plot_path = output_path / 'lstm_plot.png'
        plot_model(
            predicted=predicted,
            observed=observed,
            title=f'ROC Curve for LSTM Model, Window Size of {self.window_size}',
            df_output_path=df_path,
            plot_output_path=plot_path)

    def run(self) -> None:
        """
        Run the LSTM
        """

        x_data, y_data = self.preprocess()

        predicted, observed = self.fit_model(x_data, y_data)
        self.evaluate_model(predicted, observed)


if __name__ == '__main__':
    """
    Runs the multivariate LSTM.

    Idea: use all training variables to predict
    wandering vs. classical behavior

    """

    all_bears = unpack_data()

    # Define pipeline and run
    pipeline = MultiVarLSTM(all_bears, 15)
    pipeline.run()
