# Bears LSTMs

This subfolder contains code to model the bears' movement patterns using LSTM neural networks.

There are three modules:

## Univariate Time Series

`univariate_lstm.py` fits a model to a single time series, or one bear's movement. Here, we are concerned with only one variable - distance from water. The methodology is to predict out time steps in series of 5, i.e. use each 5 time steps to predict the next value of distance from water.

## Multivariate Classification

`multivar_lstm.py` fits a model on both male and female bears, attempting to predict the bears' wandering vs. exploratory behavior. Here, the time dependencies are ignored - we will explore this assumption later.

## Attention

`attention.py` fits a similar time series model on male and female bears to predict wandering/exploratory behavior, using an encoder-decoder attention mechanism. 

## Running

### Conda

1. Create a conda environment using `conda env create -f lstms_env.yml`
2. Run the script with `python script.py`

### Docker

If you have Docker installed, you can build and run a containerized version of whichever module you want. 

1. Build the image by running `docker build . --no-cache -t bears:1.0`
2. Run the container. For interactive execution:
    - `docker run -it bears:1.0 /bin/bash`
    - Inside the the container, just run `python <script>.py`
 For non-interactive execution, run `docker run -it bears:1.0 python <module>.py`