# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy

from ..portfolio.contract import Contract
from ..portfolio.position import Position2
from ..portfolio.portfolio import Portfolio
from ..portfolio.trade import Trade
from ..optimal_controls.ou_spread_model import (
    OUSpreadModelSolver
)
from ..simulation.simulate_cointegrated_assets import simulate_ou_spread


def compute_rebalancing_amount(target_nominal, price, contract, portfolio):
    """
    Computes number of units to rebalance
    """
    n_new = int(target_nominal / float(price * contract.multiplier))
    position = portfolio.get_position(contract.symbol)
    if position is not None:
        n_old = position.net_position
    else:
        raise ValueError(f'Cannot compute rebalancing amount. Position for symbol: {contract.symbol} not found.')

    amount = n_new - n_old

    return amount


def create_trade(price, amount, contract):
    """
    Creates a new trade
    """
    if not isinstance(price, (float, int)):
        raise ValueError('price has to be <int> or <float>.')

    if not isinstance(amount, int):
        raise ValueError('amount has to be <int>.')

    if not isinstance(contract, Contract):
        raise ValueError('contract has to be <Contract>.')

    if amount == 0:
        return None

    if amount > 0:
        action = 'BOT'
    elif amount < 0:
        action = 'SLD'
    else:
        raise ValueError('amount invalid.')

    # Compute trading commissions
    if contract.sec_type == 'STK':
        commission = (4.0 / 10000) * abs(amount) * price

    elif contract.sec_type == 'F':
        commission = 1.0

    return Trade(contract.symbol, action, price, abs(amount),
                 contract.sec_type, commission)


def simulate_pairs_trading(
        model_parameters,
        strategy_parameters,
        a, b, s, T, dt,
        contract_a, contract_b, 
        n_steps,
        simulation_number):

    #T = strategy_parameters.trading_horizon
    #dt = T / n_steps

    # Create contract objects
    #contract_a = Contract(strategy_parameters.symbol_a, 'F', 50)
    #contract_b = Contract(strategy_parameters.symbol_b, 'F', 20)

    # Simulate prices
    n_sim = 1 # hard coded, sima are instead produced by several calls.
    # a, b, s = simulate_ou_spread(
    #     n_sim, n_steps, model_parameters.b_0, model_parameters.x_0,
    #     model_parameters.kappa, model_parameters.theta,
    #     model_parameters.eta, model_parameters.mu_b,
    #     model_parameters.sigma_b, dt)

    # Create position objects
    PositionA = Position2(contract_a)
    PositionB = Position2(contract_b)
    
    # Create portfolio
    portfolio = Portfolio(f'Portfolio #{simulation_number}')
    portfolio.add_position(PositionA)
    portfolio.add_position(PositionB)

    for i in range(0, n_steps):


        #------------------------------------------------------------------#
        #                       Update Tradng Model                        #
        #------------------------------------------------------------------#
    
        # Compute ln-spread
        spread = np.log(a[i]) - np.log(b[i])

        # Percentage allocations
        time_left = T - dt*i
        optimal_decisions = OUSpreadModelSolver.solve_asset_weights(
            model_parameters, strategy_parameters, spread, time_left) 


        #------------------------------------------------------------------#
        #                      Rebalance position in A                     #
        #------------------------------------------------------------------#

        # Rebalance position in A
        amount = compute_rebalancing_amount(
            optimal_decisions.alloc_a_trunc, a[i], contract_a, portfolio)

        # Create trade
        trade = create_trade(a[i], amount, contract_a)

        if trade is not None:

            # Add the trade to the position
            portfolio.add_trade(trade)


        # Update portfolio market value
        portfolio.update_market_value(contract_a.symbol, a[i], a[i])


        #------------------------------------------------------------------#
        #                      Rebalance position in B                     #
        #------------------------------------------------------------------#

        # Rebalance position in B
        amount = compute_rebalancing_amount(
            optimal_decisions.alloc_b_trunc, b[i], contract_b, portfolio)

        # Create trade
        trade = create_trade(b[i], amount, contract_b)

        if trade is not None:

            # Add the trade to the position
            portfolio.add_trade(trade)

        

        # Update portfolio market value
        portfolio.update_market_value(contract_b.symbol, b[i], b[i])

    return portfolio


def simulate_strategy(model_parameters, strategy_parameters, n_steps, n_sim):
    

    contract_a = Contract(strategy_parameters.symbol_a, 'F', 50)
    contract_b = Contract(strategy_parameters.symbol_b, 'F', 20)
    
    portfolios = {}
    a_prices = {}
    b_prices = {}
    
    for i in range(0, n_sim):
        
        # Simulate prices and spread
        A_t, B_t, X_t = simulate_ou_spread(
             n_sim, n_steps, model_parameters.b_0, model_parameters.x_0,
             model_parameters.kappa, model_parameters.theta,
             model_parameters.eta, model_parameters.mu_b,
             model_parameters.sigma_b, 
             strategy_parameters.trading_horizon / float(n_steps))
        
        # Simulate pairs trading strategy
        portfolio = simulate_pairs_trading(
            model_parameters,
            strategy_parameters,
            A_t, B_t, X_t, 
            strategy_parameters.trading_horizon,
            strategy_parameters.trading_horizon / float(n_steps),
            contract_a, contract_b, 
            n_steps,
            i)

        portfolios.update({i: deepcopy(portfolio)})
        a_prices.update({i: A_t})
        b_prices.update({i: B_t})

    return a_prices, b_prices, portfolios
