import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data.data_handler import DataHandler
from src.estimation.coint_johansen import Johansen
from src.estimation.estimate_ou_params import estimate_ou_parameters
from src.optimal_controls.z_spread_model import (
    MultiSpreadModelParameters,
    MultiSpreadModelSolver
)

def main():

    symbols = ['BNK.PA', 'UTI.PA', 'BRE.PA'] #['MSE.PA', 'BNK.PA', 'UTI.PA', 'BRE.PA', 'HLT.PA']

    data = DataHandler.download_historical_closings(symbols).dropna()

    s_i = data
    s_0 = np.exp(np.log(data).mean(axis=1)).to_frame()
    s_0.columns = ['mean']

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(s_0, color='red')
    ax.plot(s_i, color='blue', alpha=0.5)
    plt.show()

    mu_0 = s_0.pct_change(1).dropna().mean()
    mu_i = s_i.pct_change(1).mean(axis=0).values.reshape(-1, 1)
    sigma_0 = s_0.pct_change(1).dropna().std() * np.sqrt(250)
    sigma_i = s_i.pct_change(1).dropna().std().values.reshape(-1, 1) * np.sqrt(250)

    s_ = pd.concat([s_0, s_i], axis=1)

    rho_ = s_.pct_change(1).dropna().corr().values
    rho_0 = rho_[1:, 0].reshape(-1, 1)
    rho_i = rho_[1:, 1:]

    b = np.zeros(len(sigma_i)).reshape(-1, 1)
    a = np.zeros(len(sigma_i)).reshape(-1, 1)
    betas = np.zeros(len(sigma_i)).reshape(-1, 1)
    kappas = np.zeros(len(sigma_i)).reshape(-1, 1)
    z_spreads = []
    for i, symbol in enumerate(symbols):

        # Estimate beta parameter
        y = pd.concat([np.log(s_0), np.log(s_i[symbol])], axis=1)
        estimator = Johansen(y, model=2, significance_level=0)
        e_, r = estimator.johansen()
        e = e_[:, 0] / e_[0, 0]
        betas[i] = e[1]

        # Estimate kappa parameter
        z = y.dot(e)
        z_spreads.append(pd.DataFrame(index=y.index, data=z, columns=[symbol]))
        k, theta, sigma_z = estimate_ou_parameters(z.values, 1/250)
        kappas[i] = k
        a[i] = theta

        print("The first cointegrating relation: {}".format(e))
        fig, ax = plt.subplots(2, 1, figsize=(6, 3))
        ax[0].plot(z)
        ax[1].plot(np.log(s_0), color='red')
        ax[1].plot(np.log(s_i[symbol]), color='blue', label = symbol)
        plt.legend()
        plt.show()

    z_spreads = pd.concat(z_spreads, axis=1)


    deltas = np.zeros(len(sigma_i)).reshape(-1, 1)
    for i, symbol in enumerate(symbols):
        deltas[i] = kappas[i]/(-betas[i])

    params = MultiSpreadModelParameters(-1000, rho_i, rho_0, sigma_0, sigma_i, mu_0, mu_i, betas, deltas, b, a)

    model = MultiSpreadModelSolver.solve(params, 50, 10000)

    #model.plot_b_t()
    #model.plot_c_t()

    #z_spreads.plot()
    #plt.show()

    result = {}
    for date in z_spreads.index:

        z_ = z_spreads.loc[date].values.reshape(-1, 1)

        pi = model.optimal_portfolio(z_, 10).flatten()
        pos = {}
        pos.update({symbols[0]: pi[0]})
        pos.update({symbols[1]: pi[1]})
        pos.update({symbols[2]: pi[2]})
        #pos.update({symbols[3]: pi[3]})
        result.update({date: pos})

    result = pd.DataFrame.from_dict(result, orient='index')

    pnl = data.pct_change(1) * result.shift(1)

    result.plot()
    plt.show()

    pnl.cumsum().plot()
    plt.show()

    pnl.sum(axis=1).cumsum().plot()
    plt.show()


    print(" ")

if __name__ == '__main__':
    main()