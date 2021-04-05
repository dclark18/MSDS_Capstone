import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys

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

        # Stop the model fit procedure if no loss gain in 3 epochs
        es = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        attention_model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=500,
            callbacks=[es])

        # Get predictions
        preds = attention_model.predict(x_test)

        return preds[:, 0], y_test

    def save_results(self, predicted: np.array, observed: np.array, output_dir: str) -> None:
        """
        Given labels, save the results out
        """
        df_path = Path(output_dir) / f'attention_predictions_{self.test_bear_id}.csv'
        logger.info(f"Saving results to {df_path}")

        output_df = pd.DataFrame({
            'observed': observed,
            'predicted': predicted,
            'bear_id': self.test_bear_id})

        output_df.to_csv(df_path, index=False)

    def run(self, output_dir: Path) -> None:
        """
        Run the Attention model, and save predictions
        """

        x_train, x_test, y_train, y_test = self.preprocess()

        predicted, observed = self.fit_model(x_train, x_test, y_train, y_test)

        self.save_results(predicted, observed, output_dir)


if __name__ == '__main__':

    output_path = sys.argv[1]  # Where to save outputs
    if not os.path.exists(output_path):
        # Make the directory
        logger.warning(f"{output_path} does not exist, creating")
        os.makedirs(output_path)
    logger.info(f"Outputs will be saved to {output_path}")

    # For parallel runs, use task id from SLURM array job.
    # Passed in via env variable
    try:
        test_bear_idx = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        logger.info(f"Index: {test_bear_idx}")
    except TypeError:
        # Empty variable if not run in a parallel setting (interactive mode)
        logger.warning("Non-interactive setting, index is set to 0")
        test_bear_idx = 0  # Hardcode to a random value

    all_bears = unpack_data()

    # Sort the ids and grab the index
    ids = np.sort(all_bears.Bear_ID.unique())
    test_bear_id = ids[test_bear_idx]
    logger.debug(f"Test index: {test_bear_id}")

    pipeline = AttentionModel(all_bears, 1, test_bear_id)
    pipeline.run(output_path)
