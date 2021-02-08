import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loguru import logger
from typing import Tuple

from tensorflow import keras
from keras import backend as K
from sklearn.metrics import roc_curve, auc

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
        plt.title(f'ROC Curve for Attention model, window size of {self.window_size}')
        plt.legend(loc="lower right")

        output_path = Path(os.path.abspath(__file__)).parent
        plt.savefig(os.path.join(output_path, 'multivar_roc_curve.png'))

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
