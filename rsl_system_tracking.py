import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.data_handler import (
    DataHandler
)
from src.estimation.rls import (
    RLSFilter,
    system_identification_setup
)


def download_time_series():

    symbols = ['EWA', 'EWC']

    data = DataHandler.download_historical_closings(symbols).dropna()

    return data


def test_arma_system_identification():
    """
    Here we simulate a ARMA system and seek to identify its parameters.
    """
    n_steps = 500
    dt = 1.0 / n_steps

    y_1 = np.zeros((n_steps, 1))
    for i in range(2, n_steps):
        y_1[i] = 0.01 * dt + 0.99 * y_1[i - 1] + 0.25 * np.sqrt(dt) * np.random.normal(0, 1)

    F_1 = RLSFilter(p=1, lmbda=0.9999, delta=1e-5)
    F_1 = system_identification_setup(F_1)
    w_hat_1 = np.array([F_1(y_i[0]) for y_i in y_1])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(w_hat_1)
    plt.show()

    """
    ARMA time-dependent parameters
    """
    t = np.linspace(0, 10, n_steps)
    b_t = np.sin(t)
    y_2 = np.zeros((n_steps, 1))
    for i in range(1, n_steps):
        y_2[i] = 0.01 * dt + b_t[i - 1] * y_2[i - 1] + 0.25 * np.sqrt(dt) * np.random.normal(0, 1)

    F_2 = RLSFilter(p=1, lmbda=0.90, delta=1e-6)
    F_2 = system_identification_setup(F_2)

    # Estimate and update parameters
    w_hat_2 = np.array([F_2(y_i[0]) for y_i in y_2])

    fig, ax = plt.subplots(figsize=(4, 4))
    plt.title('Time-Dependent ARMA RLSE')
    ax.plot(w_hat_2[:, 0], color='blue', ls='-', lw=2)
    ax.plot(b_t, color='red', ls=':', lw=2)
    ax.set_ylabel(r'$\beta(t)$', fontsize=15)
    plt.show()

if __name__ == '__main__':
    test_arma_system_identification()