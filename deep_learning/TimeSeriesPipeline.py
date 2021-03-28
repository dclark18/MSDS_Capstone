import pandas as pd
import numpy as np

from typing import Sequence

from sklearn.preprocessing import MinMaxScaler

from utils import window_data


class TimeSeriesPipeline:

    def __init__(self, raw_data: pd.DataFrame, window_size: int, test_bear_id: int):
        """
        Base class for a data pipeline. Inputs are raw data and the window size.

        We can define preprocessing and evaluation methods here, and subclass to
        implement model fitting
        """

        self.raw_data = raw_data
        self.window_size = window_size
        self.test_bear_id = test_bear_id

    def preprocess(self) -> Sequence[np.ndarray]:
        """
        Minimal preprocessing.

        - Creates the y variable
        - Drops columns not needed for analysis
        - memorize which bear IDs belong
          to which subsets of the data. Used for holdout later
        - splits into test and train, based on the input test bear ID
        - windows data
        - scales data

        Returns:
            xtrain, xtest, ytrain, ytest
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

        # Use single bear as holdout
        x_train = x_data.drop(self.test_bear_id)
        y_train = wandering.drop(self.test_bear_id)
        x_test = x_data.loc[self.test_bear_id]
        y_test = wandering.loc[self.test_bear_id]

        # Fit our scaler on the full training set here
        m = MinMaxScaler()
        m.fit(x_train)

        # Test can be scaled here, since it's a continuous time series
        x_test = window_data(m.transform(x_test), self.window_size)
        y_test = y_test.values[(self.window_size - 1):]

        # create stacks of windows to scale+fit the model
        # Use a loop to avoid overlapping windows from different time series
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

        return x_train_all, x_test, ytrain_all, y_test

    def fit_model(self, *args, **kwargs):
        """
        Each pipeline needs its own fit call.
        """
        raise NotImplementedError
