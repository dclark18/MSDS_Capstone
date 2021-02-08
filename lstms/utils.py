import numpy as np
import pandas as pd

import os
from pathlib import Path
from zipfile import ZipFile
from loguru import logger


def window_data(dataset: np.ndarray, window_size: int) -> np.ndarray:
    """
    Given a 2-D numpy array and a window size,
    create striding windows over the dataset.
    Args:
    - dataset: a (m, n) shaped numpy array
    - window_size: the number of obervations to window over
    Example:
    dataset = (1003, 20), window_size = 5
    Output = (998, 5, 20)
    Explanation: Starting from the 5th observation, create backwards-looking
    windows of 5 observations each.
    We cannot directly use 1st 4 observations for prediction, since we do not have
    5 previous time points
    """

    # Starting point is 0+window_size
    # Ending point is the last value

    # Create a index matrix
    # Note to DJ: Same approach as for loop + vstack, but vectorized. Idea ripped off from
    # https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
    array_idx = (
        0
        + np.expand_dims(np.arange(window_size), 0)
        + np.expand_dims(np.arange(len(dataset) - (window_size - 1)), 0).T
    )

    return dataset[array_idx]


def unpack_data() -> pd.DataFrame:
    """
    Unzips the bear data and combines male/female bears into one dataset
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

    return all_bears
