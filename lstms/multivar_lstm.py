import pandas as pd
import numpy as np

import os
from pathlib import Path
from loguru import logger
from typing import Tuple

from tensorflow import keras

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

from TimeSeriesPipeline import TimeSeriesPipeline
from utils import unpack_data


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
        lstm_model.add(keras.layers.LSTM(40))
        # lstm_model.add(keras.layers.Attention())
        lstm_model.add(keras.layers.Dense(1))

        lstm_model.compile(optimizer='adam', loss='mse')

        # Fit the model. Iterate by bear ID
        predicted, observed = super().loop_and_fit(x_data, y_data, lstm_model)
        return predicted, observed

    def evaluate_model(self, predicted: np.array, observed: np.array) -> None:
        """
        Given labels, return the accuracy and a ROC curve.
        """

        # Predicted vs. observed
        y_preds = [1 if x > 0.5 else 0 for x in predicted]
        y_true = observed
        fpr, tpr, _ = roc_curve(y_true, y_preds)
        auc_score = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for LSTM model, window size of {self.window_size}')
        plt.legend(loc="lower right")

        output_path = Path(os.path.abspath(__file__)).parent
        plt.savefig(os.path.join(output_path, 'multivar_roc_curve.png'))

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
    pipeline = MultiVarLSTM(all_bears, 5)
    pipeline.run()
