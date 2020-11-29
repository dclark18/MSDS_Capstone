# Bears LSTMs

This subfolder contains code to model the bears' movement patterns using LSTM neural networks.

There are two modules. To run:
1. Create a conda environment using `conda env create -f lstms_env.yml`
2. Run the script with `python script.py`

## Univariate Time Series

`univariate_lstm.py` fits a model to a single time series, or one bear's movement. Here, we are concerned with only one variable - distance from water. The methodology is to predict out time steps in series of 5, i.e. use each 5 time steps to predict the next value of distance from water.

## Multivariate Classification

`multivar_lstm.py` fits a model on both male and female bears, attempting to predict the bears' wandering vs. exploratory behavior. Here, the time dependencies are ignored - we will explore this assumption later.