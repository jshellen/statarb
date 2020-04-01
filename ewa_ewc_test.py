import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.data_handler import (
    DataHandler
)
from src.optimal_controls.estimation.kalman_filter import (
    kalman_filter_predict,
    kalman_filter_update
)


def download_time_series():

    symbols = ['EWA', 'EWC']

    data = DataHandler.download_historical_closings(symbols).dropna()

    return data


def main():

    data = download_time_series()

    print(" ")

if __name__ == '__main__':
    main()