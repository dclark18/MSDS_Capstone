import pandas as pd
import numpy as np

import os
import sys
from pathlib import Path
from loguru import logger
from typing import Tuple

from tensorflow import keras

from TimeSeriesPipeline import TimeSeriesPipeline
from utils import unpack_data, plot_model


class MultiVarLSTM(TimeSeriesPipeline):

    def __init__(self, raw_data: pd.DataFrame, window_size: int, test_bear_idx: int):

        super().__init__(raw_data, window_size, test_bear_idx)

    def preprocess(self) -> Tuple[np.ndarray]:
        return super().preprocess()

    def fit_model(self,
                  x_train: np.ndarray,
                  x_test: np.ndarray,
                  y_train: np.ndarray,
                  y_test: np.ndarray) -> Tuple[np.ndarray]:
        """
        Fit the multivariate LSTM, and generate predictions.

        Args:
           x_train, x_test, y_train, y_test: train test split data

        Output: arrays representing predicted and observed classifications
        """
        logger.info("Fitting the model")

        lstm_model = keras.models.Sequential()
        lstm_model.add(keras.layers.LSTM(40, stateful=False))
        lstm_model.add(keras.layers.Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')

        # Stop the model fit procedure if no loss gain in 3 epochs
        es = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        lstm_model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=20,
            callbacks=[es])

        predicted = lstm_model.predict(x_test)

        return predicted[:, 0], y_test

    def save_results(self, predicted: np.array, observed: np.array, output_path: str) -> None:
        """
        Given labels, return the accuracy and a ROC curve.
        """

        df_path = Path(output_path) / f'lstm_predictions_{self.test_bear_id}.csv'
        logger.info(f"Saving results to {df_path}")

        output_df = pd.DataFrame({
            'observed': observed,
            'predicted': predicted,
            'bear_id': self.test_bear_id})
        output_df.to_csv(df_path, index=False)

    def run(self, output_path: str) -> None:
        """
        Run the LSTM
        """

        x_train, x_test, y_train, y_test = self.preprocess()

        predicted, observed = self.fit_model(x_train, x_test, y_train, y_test)
        self.save_results(predicted, observed, output_path)


if __name__ == '__main__':
    """
    Runs the multivariate LSTM.

    Idea: use all training variables to predict
    wandering vs. classical behavior

    """

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

    pipeline = MultiVarLSTM(all_bears, 5, test_bear_id)
    pipeline.run(output_path)

