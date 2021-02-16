import pandas as pd
import numpy as np
from loguru import logger

from typing import Tuple, Any

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from utils import window_data


class TimeSeriesPipeline:

    def __init__(self, raw_data: pd.DataFrame, window_size: int):
        """
        Base class for a data pipeline. Inputs are raw data and the window size.

        We can define preprocessing and evaluation methods here, and subclass to
        implement model fitting
        """

        self.raw_data = raw_data
        self.window_size = window_size

    def preprocess(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Minimal preprocessing.

        - Creates the y variable
        - Drops columns not needed for analysis
        - memorize which bear IDs belong
          to which subsets of the data. Used for holdout later

        Returns:
            x and y data as numpy arrays
        """

        # Need to drop non-observed points
        observed = self.raw_data.loc[self.raw_data.OBSERVED == 1]

        wandering = ((observed.STEPLENGTH < 680)
                     & (np.abs(observed.TURNANGLE) > 45)).astype(int)

        # Drop the variables used to create y variable, and drop datetime
        x_data = observed.drop(
            columns=['STEPLENGTH', 'TURNANGLE', 'Unnamed: 0', 'datetime'])

        # Drop various ID columns - not real data
        x_data = x_data.drop(
            columns=['FID', 'Id', 'SAMPLEID', 'OBSERVED'])

        # Strategy for multiple time series: Split up dataset by bear,
        # and iteratively model
        # Preserve indices for training
        x_data = x_data.reset_index(drop=True).set_index('Bear_ID')
        wandering.index = x_data.index

        return x_data, wandering

    def loop_and_fit(self, x_data: pd.DataFrame, y_data: pd.Series,
                     model: Any) -> Tuple[np.ndarray]:
        """
        This splits the incoming x_data into train and test sets,
        and does n-fold validation using each bear as a holdout once.

        Args:
            - x_data: dataframe of x data to use
            - y_data: prediction variable
            - model: a compiled keras model object

        Returns:
            array of predicted labels and originals
        """

        predicted = np.array([])
        observed = np.array([])

        for holdout_id in x_data.index.unique():

            logger.info(f"Holdout bear: {holdout_id}")

            # Holdout ID is the one bear ID to be used as a holdout set
            x_test = x_data.loc[holdout_id]
            y_test = y_data.loc[holdout_id]

            # Train is everything else, but needs to be mapped to the right ID
            x_train = x_data.drop(holdout_id)
            y_train = y_data.drop(holdout_id)

            # Fit the min-max scaler here
            m = MinMaxScaler()
            m.fit(x_train)

            # Test scaled here and windowed, train scaled in inner loop
            x_test = m.transform(x_test)
            x_test = window_data(x_test, self.window_size)

            # y test vals come out as numpy array
            y_test = y_test.values[(self.window_size - 1):]

            # Inner loop: go over each remaining ID,
            # create stacks of windows to scale+fit the model
            x_train_all = None
            ytrain_all = None
            for bear_id in x_train.index.unique():

                xtrain_subset = x_train.loc[bear_id]
                ytrain_subset = y_train.loc[bear_id].values[(self.window_size - 1):]

                # Scale and window training data
                xtrain_scaled = window_data(
                    m.transform(xtrain_subset), self.window_size)

                if x_train_all is None:
                    x_train_all = xtrain_scaled
                else:
                    x_train_all = np.concatenate((x_train_all, xtrain_scaled))

                if ytrain_all is None:
                    ytrain_all = ytrain_subset
                else:
                    ytrain_all = np.concatenate((ytrain_all, ytrain_subset))

            # Finally get around to fitting the model
            model.fit(
                x_train_all,
                ytrain_all,
                validation_data=(x_test, y_test),
                epochs=20,
                shuffle=False)

            # Predict out labels, compare to observed
            predictions = model.predict(x_test)

            predicted = np.append(predicted, predictions)
            observed = np.append(observed, ytrain_subset)

            # For a new holdout set, we need to reset the model
            model.reset_states()

        return predicted, observed

    def fit_model(self, *args, **kwargs):
        """
        Each pipeline needs its own fit call.
        """
        raise NotImplementedError
