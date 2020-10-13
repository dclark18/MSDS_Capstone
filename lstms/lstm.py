import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras import layers
from sklearn.preprocess import MinMaxScaler

class LSTMPipeline:

    def __init__(self, data_path: str, bear_id: int):

        self.raw_data = pd.read_csv(data_path)
        self.bear_id = bear_id


    def preprocess(self):
        """
        Read in data, set the datetime column, and do the test/train split. 

        Returns:
        Train/test split dataset
        """

        df_formatted = self.raw_data.copy()
        df_formatted = df_formatted.loc[df_formatted.bear_id == self.bear_id, ['distrdsMIN', 'datetime']]
        df_formatted.datetime = pd.to_datetime(df_formatted.datetime)
        df_formatted = df_formatted.set_index('datetime')

        # Normalize
        m = MinMaxScaler()
        df_scaled = m.fit_transform(df_formatted)

        return df_formatted.index, df_scaled


     def fit_model(self, index, scaled_vals):

     	# Complete prototype: split naively. 
     	split_len = int(0.7*len(scaled_vals))

     	train, test = scaled_vals[:split_len], scaled_vals[split_len:]

     	


