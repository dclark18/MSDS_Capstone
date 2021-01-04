import pandas as pd
import numpy as np

import os
from pathlib import Path
from zipfile import ZipFile
from loguru import logger


from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler


def window_data(dataset: np.ndarray, window_size: int):
    """
    Given a 2-D numpy array and a window size,
    chunk the dataset into observations of length window_size

    Args:
    - dataset: a (m, n) shaped numpy array
    - window_size: the number of obervations to window over

    Example:
    dataset = (1003, 20), window_size = 5
    Output = (200, 5, 20)
    """

    # Drop the last observations that don't overlap with the window size.
    max_row = len(dataset) // window_size * window_size
    dataset = dataset[:max_row]

    # Define the number of chunks
    num_chunks = len(dataset) // window_size

    # Define n_features
    try:
        n_features = dataset.shape[1]
    except IndexError:
        n_features = 1

    data_reshaped = dataset.reshape(
        (num_chunks, window_size, n_features))
    return data_reshaped


class MultiVarLSTM:

    def __init__(self, raw_data: pd.DataFrame):

        self.raw_data = raw_data
        self.window_size = 5

    def preprocess(self):
        """
        Standard preprocessing. Define the y variable,
        trim the x variables, train/test split, and min-max scale.

        Returns:
            Scaled and split dataset for modeling.
        """
        logger.info("Starting preprocessing")
        wandering = ((self.raw_data.STEPLENGTH < 680)
                     & (np.abs(self.raw_data.TURNANGLE) > 45))

        # Hold out arbitrary bear as the test set
        test_idx = self.raw_data.loc[self.raw_data.Bear_ID == 79].index

        x_data = self.raw_data.drop(
            columns=['STEPLENGTH', 'TURNANGLE', 'Unnamed: 0', 'datetime'])

        X_train = x_data.drop(test_idx)
        y_train = wandering.drop(test_idx)

        X_test = x_data.loc[test_idx]
        y_test = wandering[test_idx]

        # Strategy for multiple time series: Split up dataset by bear,
        # and iteratively model
        # Preserve indices for training
        bear_ids = X_train.Bear_ID.unique()

        self.bear_ids_dict = {
            bearid: X_train.loc[X_train.Bear_ID == bearid].index
            for bearid in bear_ids
        }

        # Fit the minmax scaler
        m = MinMaxScaler()
        m.fit(X_train)
        # Bind to class instance for later back-transformation
        self.scaler = m

        X_train = m.transform(X_train)
        X_test = m.transform(X_test)

        # Return y values as numpy array, not pandas series
        y_train = y_train.values
        y_test = y_test.values

        # Reshape test data, since it's static.
        # Training data needs to be reshaped dynamically since shape is
        # dependent on bear.
        X_test = window_data(X_test, self.window_size)
        y_test = window_data(y_test, self.window_size)

        return X_train, X_test, y_train, y_test

    def fit_model(self,
                  x_train: np.ndarray,
                  y_train: np.ndarray,
                  x_test: np.ndarray,
                  y_test: np.ndarray):
        """
        Fit the multivariate LSTM.

        Args:
            x_train - the preprocessed training data
            y_train - the boolean classification variable
            x_test - the reshaped testing data
            y_test - the boolean reshaped testing data
        """
        logger.info("Fitting the model")

        lstm_model = keras.models.Sequential()
        lstm_model.add(keras.layers.LSTM(40))
        lstm_model.add(keras.layers.Dense(1))

        lstm_model.compile(optimizer='adam', loss='mse')

        # Fit the model. Iterate by bear ID
        for bear_id, idcs in self.bear_ids_dict.items():

            # Reshape data for sequential model
            logger.info(f"Training on bear {bear_id}")

            bear_x_data = x_train[idcs]
            bear_y_data = y_train[idcs]

            xtrain_reshaped = window_data(bear_x_data, self.window_size)
            ytrain_reshaped = window_data(bear_y_data, self.window_size)

            lstm_model.fit(xtrain_reshaped,
                           ytrain_reshaped,
                           validation_data=(x_test, y_test),
                           epochs=20)

        return lstm_model

    def predict_model(self, model, x_test, y_test):
        """
        Predict out, and get the summary statistics.
        """

        scores = model.evaluate(x_test, y_test, verbose=1)
        return scores

    def run(self):
        """
        Run the LSTM
        """

        x_train, x_test, y_train, y_test = self.preprocess()

        lstm_model = self.fit_model(x_train, y_train, x_test, y_test)
        scores = self.predict_model(lstm_model, x_test, y_test)

        return scores


if __name__ == '__main__':
    """
    Runs the multivariate LSTM.

    Idea: use all training variables to predict
    wandering vs. classical behavior

    """

    # Directory wrangling
    wd = os.path.abspath(__file__)
    top_level_path = Path(wd).parent.parent
    data_path = Path(top_level_path) / 'data'

    # Extract the male/female data if not already extracted
    if not os.path.exists(data_path):

        zip_path = Path(top_level_path) / 'data.zip'
        logger.info(f"Unpacking data files from {zip_path}")
        with ZipFile(zip_path, 'r') as zipObj:
            zipObj.extractall(top_level_path)

        logger.info(f"Data loaded to {top_level_path / 'data'}")

    # Load datasets
    male_bears = pd.read_csv(Path(data_path) / 'maleclean4.csv')
    female_bears = pd.read_csv(Path(data_path) / 'femaleclean4.csv')

    # Subset to only observed behavior, and concatenate.
    all_bears = pd.concat([male_bears, female_bears], sort=True)

    # Drop misformatted NAs
    all_bears = all_bears.loc[~(all_bears.FID.isna())]
    all_bears.reset_index(inplace=True, drop=True)

    # Define pipeline and run
    pipeline = MultiVarLSTM(all_bears)
    scores = pipeline.run()

    print(f"LSTM accuracy: {scores:.2f}")
