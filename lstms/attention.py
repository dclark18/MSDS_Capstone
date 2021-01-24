import numpy as np
import pandas as pd

from loguru import logger
from typing import Tuple

from tensorflow import keras
from keras import backend as K

from multivar_lstm import MultiVarLSTM, window_data, unpack_data


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


class AttentionModel(MultiVarLSTM):

    def __init__(self, raw_data: pd.DataFrame, window_size: int):
        """
        This class inherits from MultiVarLSTM, so we can reuse the
        preprocessing and plotting methods.
        """
        super().__init__(raw_data, window_size)

    def preprocess(self) -> Tuple[np.ndarray]:
        return super().preprocess()

    def fit_model(self,
                  x_train: np.ndarray,
                  y_train: np.ndarray,
                  x_test: np.ndarray,
                  y_test: np.ndarray) -> keras.models.Model:

        logger.info("Fitting the model")

        input_layer = keras.layers.Input((self.window_size, x_train.shape[-1]))
        expanded = keras.layers.Dense(128, activation='relu')(input_layer)
        attended = attention_simple(expanded, self.window_size)
        fc1 = keras.layers.Dense(256, activation='relu')(attended)
        fc2 = keras.layers.Dense(312, activation='relu')(fc1)
        output = keras.layers.Dense(1, activation='sigmoid')(fc2)
        attention_model = keras.models.Model(input_layer, output)

        for bear_id, idcs in self.bear_ids_dict.items():

            # Reshape data for sequential model
            logger.info(f"Training on bear {bear_id}")

            bear_x_data = x_train[idcs]
            bear_y_data = y_train[idcs]

            xtrain_reshaped = window_data(bear_x_data, self.window_size)
            # No reshaping of y, all we do is grab
            # every 5th value if the windowsize = 5
            ytrain_reshaped = bear_y_data[(self.window_size - 1):]

            attention_model.fit(xtrain_reshaped,
                                ytrain_reshaped,
                                validation_data=(x_test, y_test),
                                epochs=20)

        return attention_model

    def predict_model(self,
                      model: keras.models.Model,
                      x_test: np.ndarray,
                      y_test: np.ndarray) -> None:

        super().predict_model(model, x_test, y_test)

    def run(self) -> None:

        x_train, x_test, y_train, y_test = self.preprocess()
        attention_model = self.fit_model(x_train, x_test, y_train, y_test)

        self.predict_model(attention_model, x_test, y_test)


if __name__ == '__main__':

    all_bears = unpack_data()

    pipeline = AttentionModel(all_bears, 15)
    pipeline.run()
