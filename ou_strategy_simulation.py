# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from src.optimal_controls.ou_params import OrnsteinUhlenbeckProcessParameters
from src.optimal_controls.ou_spread_model_parameters import OUSpreadModelStrategyParameters
from src.simulation.simulate_cointegrated_assets import simulate_ou_spread
from src.simulation.simulate_pairs_trading import simulate_strategy


def main():

    # Trading strategy parameters
    nominal = 1000000
    symbol_a = 'A'
    symbol_b = 'B'
    horizon = 1
    risk_tol = -float(500)  # risk penalty parameter
    max_leverage = 1
    strategy_parameters = OUSpreadModelStrategyParameters(
        nominal, symbol_a, symbol_b,
        horizon, risk_tol, max_leverage)

    # OU process parameters
    n_sim = 20
    n_steps = 500
    b_0 = 100
    mu_b = 0.05  # drift of the asset b
    
    x_0 = 0.0
    kappa = 5.5     # spread mean-reversion speed
    theta = 0.0     # average spread level
    eta = 0.05      # spread (normal) volatility
    sigma_b = 0.20  # asset b annual volatility
    rho = 0.0       # correlation dW_x*dW_b = rho*dt, TODO: implement in simulation, curr. not supported.
    #dt = 1.0/250.0  # implied by n_steps and horizon
    model_parameters = OrnsteinUhlenbeckProcessParameters(
        kappa, theta, eta, sigma_b, rho, mu_b, x_0, b_0)
    
    # Run strategy simulation
    a_prices, b_prices, portfolios = simulate_strategy(
        model_parameters, strategy_parameters, n_steps, n_sim)

    # Plot results
    pos_a = portfolios[0].get_position('A')
    pos_b = portfolios[0].get_position('B')

    report_a = pos_a.generate_report_frame()
    report_b = pos_b.generate_report_frame()

    fig, ax = plt.subplots(3, 1, figsize=(8, 6))
    
    # Plot asset prices
    ax[0].plot(a_prices[0], color='red', label='A price')
    ax[0].plot(b_prices[0], color='blue', label='B price')
    ax[0].legend(loc=2)
    
    # Plot logarithmic spread
    ax02 = ax[0].twinx()
    ax02.plot(np.log(a_prices[0]) - np.log(b_prices[0]), color='black', label='ln(A)-ln(B)')
    ax02.legend(loc=1)

    # Plot positions
    ax[1].plot(report_a['NET_POSITION'], color='red', label='A')
    ax[1].plot(report_b['NET_POSITION'], color='blue', label='B')
    ax[1].set_ylabel('Positions')
    
    # Plot profit and loss 
    ax[2].plot(report_a['TOTAL_PNL'], color='red', label='A')
    ax[2].plot(report_b['TOTAL_PNL'], color='blue', label='B')
    ax[2].legend(loc=2)

    ax22 = ax[2].twinx()
    ax22.plot(report_a['TOTAL_PNL'] + report_b['TOTAL_PNL'], color='black', label='A+B')
    ax22.legend(loc=1)

    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    for k, v in portfolios.items():

        pos_a = v.get_position('A')
        pos_b = v.get_position('B')
        report_a = pos_a.generate_report_frame()
        report_b = pos_b.generate_report_frame()
        ax.plot(report_a['TOTAL_PNL'] + report_b['TOTAL_PNL'],
                color='blue', label='A+B', alpha=0.1)

    plt.show()

    #plot_optimal_solution(X, ou_params, model_params, 100)

    print(" ")

if __name__ == '__main__':
    main()


