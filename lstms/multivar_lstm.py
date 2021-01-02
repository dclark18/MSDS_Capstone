import pandas as pd
import numpy as np

import os
from pathlib import Path
from zipfile import ZipFile
from loguru import logger


from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


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

        x_data = self.raw_data.drop(
            columns=['STEPLENGTH', 'TURNANGLE', 'Unnamed: 0', 'datetime'])

        # Create validation/testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            x_data, wandering, test_size=0.2, random_state=42)

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

        # Reshape data for sequential model
        train_shape = len(X_train) // self.window_size, self.window_size
        test_shape = len(X_test) // self.window_size, self.window_size

        X_train = X_train.reshape((*train_shape, X_train.shape[1]))
        X_test = X_test.reshape((*test_shape, X_test.shape[1]))
        y_train = y_train.reshape((*train_shape, 1))
        y_test = y_test.reshape((*test_shape, 1))

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
        """
        logger.info("Fitting the model")

        lstm_model = keras.models.Sequential()
        lstm_model.add(keras.layers.LSTM(40))
        lstm_model.add(keras.layers.Dense(1))

        lstm_model.compile(optimizer='adam', loss='mse')

        # Fit the model
        lstm_model.fit(x_train,
                       y_train,
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
    all_bears = pd.concat([male_bears, female_bears])
    all_bears = all_bears.loc[all_bears.OBSERVED == 1]

    # Temp: Sample only a couple rows to prevent blowing up RAM
    np.random.seed(10)
    random_rows = np.random.randint(low=0, high=len(all_bears), size=500)
    logger.info("Subsetting 500 rows from data for flexilibity")
    all_bears = all_bears.iloc[random_rows]

    # Define pipeline and run
    pipeline = MultiVarLSTM(all_bears)
    scores = pipeline.run()

    print(f"LSTM accuracy: {scores:.2f}")
