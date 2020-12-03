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

    def preprocess(self):
        """
        Standard preprocessing. Define the y variable,
        trim the x variables, train/test split, and min-max scale.

        Returns:
            Scaled and split dataset for modeling.
        """

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

        return X_train, X_test, y_train, y_test

    def fit_model(self,
                  x_train: np.ndarray,
                  y_train: pd.Series):
        """
        Fit the multivariate LSTM.

        Args:
            x_train - the preprocessed training data
            y_train - the boolean classification variable
        """

        n_features = x_train.shape[1]

        lstm_model = keras.layers.Sequential()

        lstm_model.add(keras.layers.LSTM(40,
                                         activation='relu',
                                         input_shape=(None, n_features),
                                         stateful=False))
        lstm_model.add(keras.layers.Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')

        # Fit the model
        lstm_model.fit(x_train, epochs=20)
        return lstm_model

    def run(self):
        """
        Run the LSTM
        """

        x_train, x_test, y_train, y_test = self.preprocess()


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
    dataset = pipeline.preprocess()
    breakpoint()
