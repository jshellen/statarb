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
from ..simulation.simulate_cointegrated_assets import simulate_cointegrated_assets


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


def simulate_pairs_trading(model_parameters, strategy_parameters, n_steps):

    # Create contract objects
    contract_a = Contract(strategy_parameters.symbol_a, 'F', 50)
    contract_b = Contract(strategy_parameters.symbol_b, 'F', 20)

    # Simulate prices
    a, b, s = simulate_cointegrated_assets(1, n_steps, 100, 0,
                                           model_parameters.kappa, model_parameters.theta,
                                           model_parameters.eta, model_parameters.sigma_b,
                                           1.0/250.0)

    # Create position objects
    PositionA = Position2(contract_a)
    PositionB = Position2(contract_b)
    
    # Create portfolio
    portfolio = Portfolio('Test')
    portfolio.add_position(PositionA)
    portfolio.add_position(PositionB)

    for i in range(0, n_steps):

        # Compute ln-spread
        spread = (np.log(a[i]) - np.log(b[i]))[0]

        # Percentage allocations
        optimal_decisions = OUSpreadModelSolver.solve_asset_weights(model_parameters,
                                                                    strategy_parameters, spread, 1)


        # Rebalance position in A
        amount = compute_rebalancing_amount(optimal_decisions.alloc_a_trunc,
                                            a[i], contract_a, portfolio)

        # Create trade
        trade = create_trade(a[i][0], amount, contract_a)

        if trade is not None:

            # Add the trade to the position
            portfolio.add_trade(trade)


        # Update portfolio market value
        portfolio.update_market_value(contract_a.symbol, a[i][0], a[i][0])


        # Rebalance position in B
        amount = compute_rebalancing_amount(optimal_decisions.alloc_b_trunc,
                                            b[i], contract_b, portfolio)

        # Create trade
        trade = create_trade(b[i][0], amount, contract_b)

        if trade is not None:

            # Add the trade to the position
            portfolio.add_trade(trade)

        # Update portfolio market value
        portfolio.update_market_value(contract_b.symbol, b[i][0], b[i][0])

    return a, b, s, portfolio


def simulate_strategy(model_parameters, strategy_parameters, n_steps, n_sim):

    portfolios = {}
    a_prices = {}
    b_prices = {}

    for i in range(0, n_sim):

        a, b, s, portfolio = simulate_pairs_trading(model_parameters, strategy_parameters, n_steps)

        portfolios.update({i: deepcopy(portfolio)})
        a_prices.update({i: a})
        b_prices.update({i: b})

    return a_prices, b_prices, portfolios
