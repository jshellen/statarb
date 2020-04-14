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


def run_rolling_estimation(name, ln_s_0, ln_s_i):

    if not isinstance(ln_s_0, pd.core.frame.DataFrame):
        raise TypeError('ln_s_0 should be <pd.core.frame.DataFrame>.')

    if not isinstance(ln_s_i, pd.core.frame.DataFrame):
        raise TypeError('ln_s_i should be <pd.core.frame.DataFrame>.')

    if name not in list(ln_s_i.columns):
        raise ValueError(f'{name} not in ln_s_i.')

    # Take common index
    common_index = ln_s_0.index.intersection(ln_s_i[name].index)

    coefs = {}
    for i in range(250, len(common_index)):

        ix = common_index[i]
        coef = estimate_model_parameters(
            ln_s_0.loc[:common_index[i]], ln_s_i[name].loc[:common_index[i]]
        )

        coefs.update({str(ix): coef})

    file_name = f'C://Users//Juhis//Desktop//db//{str(name).replace(".", "_")}.json'
    with open(file_name, 'w') as fp:
        json.dump(coefs, fp)


def load_cointegration_coefficients(file_name):

    file_name = f'C://Users//Juhis//Desktop//db//{str(file_name).replace(".", "_")}.json'
    with open(file_name, 'r') as f:
        data = json.load(f)

    return data


def main():

    ln_s_0 = construct_ln_s_0_time_series()
    ln_s_i = construct_ln_s_i_time_series()

    pd.DataFrame.from_dict(load_cointegration_coefficients('ABI.BR'), orient='index').plot()
    pd.DataFrame.from_dict(load_cointegration_coefficients('AD.AS'), orient='index').plot()
    pd.DataFrame.from_dict(load_cointegration_coefficients('ADS.DE'), orient='index').plot()
    pd.DataFrame.from_dict(load_cointegration_coefficients('AI.PA'), orient='index').plot()
    pd.DataFrame.from_dict(load_cointegration_coefficients('AIR.PA'), orient='index').plot()
    pd.DataFrame.from_dict(load_cointegration_coefficients('ALV.DE'), orient='index').plot()
    plt.show()

    # run_rolling_estimation('ABI.BR', ln_s_0, ln_s_i)
    # run_rolling_estimation('AD.AS', ln_s_0, ln_s_i)
    # run_rolling_estimation('ADS.DE', ln_s_0, ln_s_i)
    # run_rolling_estimation('AI.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('AIR.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('ALV.DE', ln_s_0, ln_s_i)
    # run_rolling_estimation('ASML.AS', ln_s_0, ln_s_i)
    # run_rolling_estimation('BAS.DE', ln_s_0, ln_s_i)
    # run_rolling_estimation('BAYN.DE', ln_s_0, ln_s_i)
    # run_rolling_estimation('BBVA.MC', ln_s_0, ln_s_i)
    # run_rolling_estimation('BMW.DE', ln_s_0, ln_s_i)
    # run_rolling_estimation('BN.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('BNP.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('CRG.IR', ln_s_0, ln_s_i)
    # run_rolling_estimation('CS.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('DAI.DE', ln_s_0, ln_s_i)
    # run_rolling_estimation('DG.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('DPW.DE', ln_s_0, ln_s_i)
    # run_rolling_estimation('DTE.DE', ln_s_0, ln_s_i)
    # run_rolling_estimation('EL.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('ENEL.MI', ln_s_0, ln_s_i)
    # run_rolling_estimation('ENGI.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('ENI.MI', ln_s_0, ln_s_i)
    # run_rolling_estimation('EOAN.DE', ln_s_0, ln_s_i)
    # run_rolling_estimation('FP.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('FRE.DE', ln_s_0, ln_s_i)
    # run_rolling_estimation('GLE.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('IBE.MC', ln_s_0, ln_s_i)
    # run_rolling_estimation('INGA.AS', ln_s_0, ln_s_i)
    # run_rolling_estimation('ISP.MI', ln_s_0, ln_s_i)
    # run_rolling_estimation('ITX.MC', ln_s_0, ln_s_i)
    # run_rolling_estimation('LIN.DE', ln_s_0, ln_s_i)
    # run_rolling_estimation('MC.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('MUV2.DE', ln_s_0, ln_s_i)
    # run_rolling_estimation('NOKIA.HE', ln_s_0, ln_s_i)
    # run_rolling_estimation('OR.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('ORA.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('PHIA.AS', ln_s_0, ln_s_i)
    # run_rolling_estimation('SAF.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('SAN.MC', ln_s_0, ln_s_i)
    # run_rolling_estimation('SAN.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('SAP.DE', ln_s_0, ln_s_i)
    # run_rolling_estimation('SGO.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('SIE.DE', ln_s_0, ln_s_i)
    # run_rolling_estimation('SU.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('TEF.MC', ln_s_0, ln_s_i)
    # run_rolling_estimation('UNA.AS', ln_s_0, ln_s_i)
    # run_rolling_estimation('URW.AS', ln_s_0, ln_s_i)
    # run_rolling_estimation('VIV.PA', ln_s_0, ln_s_i)
    # run_rolling_estimation('VOW.DE', ln_s_0, ln_s_i)



    #common_index = ln_s_0.index.intersection(ln_s_i.index)
    #ln_s_0 = ln_s_0.loc[common_index]
    #ln_s_i = ln_s_i.loc[common_index]




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