import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.data_handler import (
    DataHandler
)
from src.optimal_controls.z_spread_model_parameters import (
    ZSpreadModelParameters
)
from src.optimal_controls.z_spread_model_solver import (
    ZSpreadModelSolver
)


def construct_ln_s_i_time_series():

    symbols = list(pd.read_excel('symbols.xlsx', sheet_name='STOXX50', header=0).values.flatten())

    data = DataHandler.download_historical_closings(symbols).dropna()

    ln_s_i = np.log(data)

    return ln_s_i


def construct_ln_s_0_time_series():

    data = DataHandler.download_historical_closings('EXW1.DE').dropna()

    ln_s_0 = np.log(data)

    return ln_s_0


def estimate_model_parameters(ln_s_0, ln_s_i):

    common_index = ln_s_0.index.intersection(ln_s_i.index)
    n_splits = 10
    kappas = np.zeros((ln_s_i.shape[1], n_splits))
    kf = KFold(n_splits=n_splits)

    for i, (_, split) in enumerate(kf.split(common_index)):
        ln_s_0_ = ln_s_0.loc[common_index[split]].values
        ln_s_i_ = ln_s_i.loc[common_index[split]].values

        params = ZSpreadModelParameters.estimate_from_ln_prices(
            ln_s_0_, ln_s_i_, gamma=-2, kappa_min=1.5)

    return params


def main():

    ln_s_0 = construct_ln_s_0_time_series()
    ln_s_i = construct_ln_s_i_time_series()
    common_index = ln_s_0.index.intersection(ln_s_i.index)
    ln_s_0 = ln_s_0.loc[common_index]
    ln_s_i = ln_s_i.loc[common_index]

    params = ZSpreadModelParameters.estimate_from_ln_prices(ln_s_0.tail(500), ln_s_i.tail(500), gamma=-2, kappa_min=1.5)

    model = ZSpreadModelSolver.solve(params, 50, 50000)

    holding = np.zeros((len(ln_s_0), len(params.m_symbols)))
    z = np.zeros((len(ln_s_0), len(params.m_symbols)))
    for i in range(0, len(ln_s_0)):
        z_t = np.zeros((len(params.m_symbols), 1))
        for j in range(0, len(params.m_symbols)):
            z_t[j, 0] = params.m_a[j] + ln_s_0.iloc[i] + params.m_beta[j] * ln_s_i[params.m_symbols[j]].iloc[i]
        pi = model.optimal_portfolio(z_t, 25)
        pi = pi/np.linalg.norm(abs(pi), 2)
        z[i, :] = z_t.reshape(1, -1)
        holding[i, :] = pi.reshape(1, -1)

    dln_s_i = ln_s_i[params.m_symbols].diff(1).dropna().values
    pnl = holding[0:-1, :] * dln_s_i

    fig, ax = plt.subplots(4, 1, figsize=(8, 6))
    ax[0].plot(ln_s_i, color='blue')
    ax[0].plot(ln_s_0, color='red')
    ax[1].plot(z[:, 0])
    ax[1].plot(z[:, 1])
    ax[1].plot(z[:, 2])

    ax[2].plot(holding)
    ax[3].plot(np.cumsum(pnl, axis=0))
    ax3_2 = ax[3].twinx()
    ax3_2.plot(np.sum(np.cumsum(pnl, axis=0), axis=1), color='black')
    plt.show()

    print(" ")

if __name__ == '__main__':
    main()