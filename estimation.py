import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data.data_handler import DataHandler

from src.optimal_controls.z_spread_model_parameters import  (
    ZSpreadModelParameters
)

from src.optimal_controls.z_spread_model_solver import (
    ZSpreadModelSolver
)

tickers = {
    'iShares Edge MSCI Europe Value Factor UCITS ETF EUR': 'IEVL.L',
    'iShares Core DAX UCITS ETF (DE)': 'EXS1.DE',
    'iShares TecDAX UCITS ETF (DE)': 'EXS2.DE',
    'iShares MDAX UCITS ETF (DE)': 'EXS3.DE',
    'iShares MSCI AC Far East ex-Japan UCITS ETF': 'IQQF.DE',
    'iShares STOXX Europe 600 UCITS ETF': 'EXSA.DE',
    'Amundi MSCI Emerging Markets UCITS ETF-C ': 'AMEM.DE'


}

def main():

    #symbols = list(tickers.values())
    symbols = ['IQQF.DE', 'AMEM.DE']

    data = DataHandler.download_historical_closings(symbols).dropna()

    data = data.loc[data.index.year > 2016]

    ln_s_0 = np.log(data).mean(axis=1).values
    ln_s_i = np.log(data).values


    params = ZSpreadModelParameters.estimate_from_ln_prices(ln_s_0[100:-100], ln_s_i[100:-100, :], gamma=-2)

    model = ZSpreadModelSolver.solve(params, 50, 20000)


    holding = np.zeros_like(ln_s_i)
    z = np.zeros_like(ln_s_i)
    for i in range(0, len(data)):
        z_t = np.zeros((len(symbols), 1))
        for j in range(0, len(symbols)):
            z_t[j, 0] = params.m_a[j] + ln_s_0[i] + params.m_beta[j] * ln_s_i[i, j]
        pi = model.optimal_portfolio(z_t, 25)
        pi = pi/np.linalg.norm(abs(pi), 2)
        z[i, :] = z_t.reshape(1, -1)
        holding[i, :] = pi.reshape(1, -1)

    dln_s_i = np.diff(ln_s_i, 1, axis=0)
    pnl = holding[0:-1, :] * dln_s_i

    fig, ax = plt.subplots(4, 1, figsize=(8, 6))
    ax[0].plot(ln_s_i, color='blue')
    ax[0].plot(ln_s_0, color='red')
    ax[1].plot(z[:, 0])
    ax[2].plot(holding)
    ax[3].plot(np.cumsum(pnl, axis=0))
    ax3_2 = ax[3].twinx()
    ax3_2.plot(np.sum(np.cumsum(pnl, axis=0), axis=1), color='black')
    plt.show()

    print(" ")

if __name__ == '__main__':
    main()