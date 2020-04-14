import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

from src.data.data_handler import (
    DataHandler
)
from src.estimation.coint_johansen import (
    Johansen
)


def construct_ln_s_i_close_series():

    # Load symbols from a file
    symbols = list(pd.read_excel('symbols.xlsx',
                                 sheet_name='STOXX50', header=0).values.flatten())

    # Download data
    symbol_data = DataHandler.download_historical_closings(symbols)

    # Collect closing prices
    ln_s_i = {}
    for symbol, data in symbol_data.items():
        frame = pd.DataFrame.from_dict(data['Close'], orient='index')
        frame.columns = [symbol]
        frame = np.log(frame)
        ln_s_i.update({symbol: frame.copy()})

    return ln_s_i


def construct_ln_s_0_close_series():

    data = DataHandler.download_historical_closings('EXW1.DE')

    ln_s_0 = np.log(pd.DataFrame.from_dict(
        data['EXW1.DE']['Close'], orient='index'))
    ln_s_0.columns = ['EXW1.DE']

    return ln_s_0


def estimate_model_parameters(ln_s_0, ln_s_i):

    if not isinstance(ln_s_0, (pd.core.frame.DataFrame,
                               pd.core.series.Series)):
        raise TypeError(f'ln_s_0 should be <pd.core.frame.DataFrame>. Input was {type(ln_s_0)}')

    if not isinstance(ln_s_i, (pd.core.frame.DataFrame,
                               pd.core.series.Series)):
        raise TypeError(f'ln_s_i should be <pd.core.frame.DataFrame>. Input was {type(ln_s_i)}')

    # Take common index
    common_index = ln_s_0.index.intersection(ln_s_i.index)

    # Estimate cointegration relationships
    x = pd.concat([ln_s_0, ln_s_i], axis=1).dropna()

    result = None
    if len(x) > 0:
        estimator = Johansen(x, model=2, significance_level=0)
        e_, r = estimator.johansen()
        e = e_[:, 0] / e_[0, 0]
        result = e[1]

    return result


def run_rolling_estimation(ln_s_0, ln_s_i):

    if not isinstance(ln_s_0, (pd.core.frame.DataFrame,
                               pd.core.series.Series)):
        raise TypeError('ln_s_0 should be <pd.core.frame.DataFrame>.')

    if not isinstance(ln_s_i, (pd.core.frame.DataFrame,
                               pd.core.series.Series)):
        raise TypeError('ln_s_i should be <pd.core.frame.DataFrame>.')

    y = ln_s_0.dropna()
    x = ln_s_i.dropna()

    # Take common index
    common_index = y.index.intersection(x.index)

    if len(common_index) < 250:
        raise ValueError('Number of overlapping observations < 250')

    y = ln_s_0.loc[common_index]
    x = ln_s_i.loc[common_index]

    x.plot()
    plt.show()
    coefs = {}
    for i in range(250, len(common_index)):

        ix = common_index[i]
        coef = estimate_model_parameters(y.iloc[:i], x.iloc[:i])

        coefs.update({str(ix): coef})

    file_name = f'C://Users//juhhel//Desktop//db//{str(ln_s_i.columns[0]).replace(".", "_")}.json'
    with open(file_name, 'w') as fp:
        json.dump(coefs, fp)


def load_cointegration_coefficients(file_name):

    file_name = f'C://Users//juhhel//Desktop//db//{str(file_name).replace(".", "_")}.json'
    with open(file_name, 'r') as f:
        data = json.load(f)

    return data


def main():

    #pd.DataFrame.from_dict(load_cointegration_coefficients('ALV.DE'), orient='index').plot()
    #pd.DataFrame.from_dict(load_cointegration_coefficients('ASML.AS'), orient='index').plot()
    #plt.show()

    ln_s_0 = construct_ln_s_0_close_series()
    ln_s_i = construct_ln_s_i_close_series()

    #pd.DataFrame.from_dict(load_cointegration_coefficients('ABI.BR'), orient='index').plot()
    #pd.DataFrame.from_dict(load_cointegration_coefficients('AD.AS'), orient='index').plot()
    #pd.DataFrame.from_dict(load_cointegration_coefficients('ADS.DE'), orient='index').plot()
    #pd.DataFrame.from_dict(load_cointegration_coefficients('AI.PA'), orient='index').plot()
    #pd.DataFrame.from_dict(load_cointegration_coefficients('AIR.PA'), orient='index').plot()

    #plt.show()

    run_rolling_estimation(ln_s_0, ln_s_i['ABI.BR'])
    run_rolling_estimation(ln_s_0, ln_s_i['AD.AS'])
    run_rolling_estimation(ln_s_0, ln_s_i['ADS.DE'])
    run_rolling_estimation(ln_s_0, ln_s_i['AI.PA'])
    run_rolling_estimation(ln_s_0, ln_s_i['AIR.PA'])
    run_rolling_estimation(ln_s_0, ln_s_i['ALV.DE'])
    run_rolling_estimation(ln_s_0, ln_s_i['ASML.AS'])
    run_rolling_estimation(ln_s_0, ln_s_i['BAS.DE'])
    run_rolling_estimation(ln_s_0, ln_s_i['BAYN.DE'])
    run_rolling_estimation(ln_s_0, ln_s_i['BBVA.MC'])
    run_rolling_estimation(ln_s_0, ln_s_i['BMW.DE'])
    run_rolling_estimation(ln_s_0, ln_s_i['BN.PA'])
    run_rolling_estimation(ln_s_0, ln_s_i['BNP.PA'])
    run_rolling_estimation(ln_s_0, ln_s_i['CRG.IR'])
    run_rolling_estimation(ln_s_0, ln_s_i['CS.PA'])
    run_rolling_estimation(ln_s_0, ln_s_i['DAI.DE'])
    run_rolling_estimation(ln_s_0, ln_s_i['DG.PA'])
    run_rolling_estimation(ln_s_0, ln_s_i['DPW.DE'])
    run_rolling_estimation(ln_s_0, ln_s_i['DTE.DE'])






    # params = ZSpreadModelParameters.estimate_from_ln_prices(ln_s_0.tail(500), ln_s_i.tail(500), gamma=-2, kappa_min=1.5)
    #
    # model = ZSpreadModelSolver.solve(params, 50, 50000)
    #
    # holding = np.zeros((len(ln_s_0), len(params.m_symbols)))
    # z = np.zeros((len(ln_s_0), len(params.m_symbols)))
    # for i in range(0, len(ln_s_0)):
    #     z_t = np.zeros((len(params.m_symbols), 1))
    #     for j in range(0, len(params.m_symbols)):
    #         z_t[j, 0] = params.m_a[j] + ln_s_0.iloc[i] + params.m_beta[j] * ln_s_i[params.m_symbols[j]].iloc[i]
    #     pi = model.optimal_portfolio(z_t, 25)
    #     pi = pi/np.linalg.norm(abs(pi), 2)
    #     z[i, :] = z_t.reshape(1, -1)
    #     holding[i, :] = pi.reshape(1, -1)
    #
    # dln_s_i = ln_s_i[params.m_symbols].diff(1).dropna().values
    # pnl = holding[0:-1, :] * dln_s_i
    #
    # fig, ax = plt.subplots(4, 1, figsize=(8, 6))
    # ax[0].plot(ln_s_i, color='blue')
    # ax[0].plot(ln_s_0, color='red')
    # ax[1].plot(z[:, 0])
    # ax[1].plot(z[:, 1])
    # ax[1].plot(z[:, 2])
    #
    # ax[2].plot(holding)
    # ax[3].plot(np.cumsum(pnl, axis=0))
    # ax3_2 = ax[3].twinx()
    # ax3_2.plot(np.sum(np.cumsum(pnl, axis=0), axis=1), color='black')
    # plt.show()
    #
    print(" ")

if __name__ == '__main__':
    main()