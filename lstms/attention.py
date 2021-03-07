import numpy as np
import pandas as pd
from pathlib import Path
import os

from loguru import logger
from typing import Tuple

from tensorflow import keras
from keras import backend as K

from TimeSeriesPipeline import TimeSeriesPipeline
from utils import unpack_data, plot_model


def attention_simple(inputs: np.ndarray, n_time: int) -> keras.layers.Lambda:
    """
    Define our own attention layer. From DJ

    Args:
    inputs - training data to be input
    n_time - length of the time dimension
    """
    input_dim = int(inputs.shape[-1])
    a = keras.layers.Permute((2, 1), name='temporalize')(inputs)  # (60,16)
    a = keras.layers.Dense(n_time,
                           activation='softmax', name='attention_probs')(a)  # (60 16)
    a_probs = keras.layers.Permute((2, 1), name='attention_vec')(a)  # (16 60)
    x = keras.layers.Dense(input_dim, activation=None, name='identity')(inputs)
    output_mul = keras.layers.Multiply(name='focused_attention')([x, a_probs])  # (16 60)
    output_flat = keras.layers.Lambda(
        lambda x: K.sum(x, axis=1), name='temporal_average')(output_mul)
    return output_flat


class AttentionModel(TimeSeriesPipeline):

    def __init__(self, raw_data: pd.DataFrame, window_size: int):
        """
        This class inherits from MultiVarLSTM, so we can reuse the
        preprocessing and plotting methods.
        """
        super().__init__(raw_data, window_size)

    def preprocess(self) -> Tuple[pd.DataFrame, pd.Series]:
        return super().preprocess()

    def fit_model(self,
                  x_data: pd.DataFrame,
                  y_data: pd.Series) -> Tuple[np.ndarray]:
        """
        Define and fit an attention model, and return the labels.
        """

        logger.info("Fitting the model")

        # Defines the attention model
        input_layer = keras.layers.Input((self.window_size, x_data.shape[-1]))
        expanded = keras.layers.Dense(128, activation='relu')(input_layer)
        attended = attention_simple(expanded, self.window_size)
        fc1 = keras.layers.Dense(256, activation='relu')(attended)
        fc2 = keras.layers.Dense(312, activation='relu')(fc1)
        output = keras.layers.Dense(1, activation='sigmoid')(fc2)
        attention_model = keras.models.Model(input_layer, output)
        attention_model.compile(optimizer='adam', loss='mse')

        predicted, observed = super().loop_and_fit(x_data, y_data, attention_model)

        return predicted, observed

    def evaluate_model(self, predicted: np.array, observed: np.array) -> None:
        """
        Given labels, return the accuracy and a ROC curve.
        """
        output_path = Path(os.path.abspath(__file__)).parent
        df_path = output_path / 'attention_predictions.csv'
        plot_path = output_path / 'attention_plot.png'
        plot_model(
            predicted=predicted,
            observed=observed,
            title=f'ROC Curve for Attention Model, Window Size of {self.window_size}',
            df_output_path=df_path,
            plot_output_path=plot_path)

    def run(self) -> None:
        """
        Run the Attention model
        """

        x_data, y_data = self.preprocess()

        predicted, observed = self.fit_model(x_data, y_data)
        self.evaluate_model(predicted, observed)


if __name__ == '__main__':

    all_bears = unpack_data()

    pipeline = AttentionModel(all_bears, 15)
    pipeline.run()
