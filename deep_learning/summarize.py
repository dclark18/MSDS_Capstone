import argparse
import pandas as pd
from matplotlib import pyplot as plt
import glob


def pull_data(output_dir: str):

    data_files = glob.glob(f"{output_dir}/*.csv")

    df = pd.concat((pd.read_csv(f) for f in data_files), sort=True)
    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path")

    args = parser.parse_args()

    output_path = args.output_path  # Root of directory to read in inputs from
    df = pull_data(output_path)
