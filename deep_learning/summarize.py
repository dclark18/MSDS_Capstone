import argparse
import pandas as pd
from utils import plot_model
import glob
from pathlib import Path


def pull_data(output_dir: str) -> pd.DataFrame:
    """ Pulls and concatenates separate csvs from model run"""
    data_files = glob.glob(f"{output_dir}/*.csv")
    df = pd.concat((pd.read_csv(f) for f in data_files), sort=True)
    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path")
    parser.add_argument("--title")

    args = parser.parse_args()

    output_path = args.output_path  # Root of directory to read in inputs from
    title = args.title
    df = pull_data(output_path)

    plot_model(
        predicted=df.predicted,
        observed=df.observed,
        title=title,
        plot_output_path=Path(output_path) / f"diagnostics/{title}.png")

    # Save compiled data frame
    df.to_csv(Path(output_path) / "diagnostics/summarized.csv")
