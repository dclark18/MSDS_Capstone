import numpy as np
import pandas as pd
from pathlib import Path
import os

from loguru import logger
from typing import Tuple

from tensorflow import keras
from keras import backend as K

from TimeSeriesPipeline import TimeSeriesPipeline
from utils import unpack_data


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

    def __init__(self, raw_data: pd.DataFrame, window_size: int, test_bear_id: int):
        """
        This class inherits from TimeSeriesPipeline, so we can reuse the
        preprocessing and plotting methods.
        """
        super().__init__(raw_data, window_size, test_bear_id)

    def preprocess(self) -> Tuple[np.ndarray]:
        return super().preprocess()

    def fit_model(self,
                  x_train: np.ndarray,
                  x_test: np.ndarray,
                  y_train: np.ndarray,
                  y_test: np.ndarray) -> Tuple[np.ndarray]:
        """
        Define and fit an attention model, and return the labels.
        """

        logger.info("Fitting the model")

        # Defines the attention model
        input_layer = keras.layers.Input((self.window_size, x_train.shape[-1]))
        expanded = keras.layers.Dense(128, activation='relu')(input_layer)
        attended = attention_simple(expanded, self.window_size)
        fc1 = keras.layers.Dense(256, activation='relu')(attended)
        fc2 = keras.layers.Dense(312, activation='relu')(fc1)
        output = keras.layers.Dense(1, activation='sigmoid')(fc2)
        attention_model = keras.models.Model(input_layer, output)
        attention_model.compile(optimizer='adam', loss='mse')

        attention_model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=20)

        # Get predictions
        preds = attention_model.predict(x_test)

        return preds, y_test

    def save_results(self, predicted: np.array, observed: np.array) -> None:
        """
        Given labels, return the accuracy and a ROC curve.
        """
        output_path = Path(os.path.abspath(__file__)).parent
        df_path = output_path / f'attention_predictions_{self.test_bear_id}.csv'
        logger.info(f"Saving results to {df_path}")

        # plot_model(
        #     predicted=predicted,
        #     observed=observed,
        #     title=f'ROC Curve for Attention Model, Window Size of {self.window_size}',
        #     df_output_path=df_path,
        #     plot_output_path=plot_path)

    def run(self, output_dir: Path) -> None:
        """
        Run the Attention model, and save predictions
        """

        x_data, y_data = self.preprocess()

        predicted, observed = self.fit_model(x_data, y_data)

        self.save_results(predicted, observed)


if __name__ == '__main__':

    # For parallel runs, use task id from SLURM array job.
    # Passed in via env variable
    test_bear_idx = int(os.getenv("SLURM_ARRAY_JOB_ID"))

    if not test_bear_idx:
        # Empty variable if not run in a parallel setting (interactive mode)
        test_bear_idx = 0  # Hardcode to a random value

    all_bears = unpack_data()

    # Sort the ids and grab the index
    ids = np.sort(all_bears.Bear_ID.unique())
    test_bear_id = ids[test_bear_idx]
    logger.debug(f"Test index: {test_bear_id}")

    pipeline = AttentionModel(all_bears, 15, test_bear_id)
    pipeline.run()
