# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:49:32 2019

@author: helleju
"""

from ..data.data_request import BloombergAPI
from ..portfolio.contract import Contract
from ..portfolio.position import Position
from ..portfolio.portfolio import Portfolio
from ..portfolio.trade import Trade
from ..optimal_controls.HJBsolutions import OUParameters, HJB_Spread_Allocator, AllocationParameters

import numpy as np
import pandas
from datetime import datetime

def simulate_pairs_trading(symbols,pair_settings):
    
    api          = BloombergAPI()
    start_date   = datetime(2000,1,1)
    end_date     = datetime.today()
    
    price_data   = []
    for symbol in symbols:
        print(symbol)
        try:
            data = api.send_daily_data_request([symbol],["PX_LAST"],start_date,end_date)
            if(len(data[symbol])!=0):
                f = pandas.DataFrame.from_dict(data[symbol],orient='index')
                f.columns = [symbol]
                price_data.append(f.copy())
        except:
            print(f"Could not download data for {symbol}")
    
    price_data       = pandas.concat(price_data,axis=1)
    price_data.index = pandas.DatetimeIndex(price_data.index)
    price_data       = price_data.dropna()
    
    
    end_date  = price_data.index[-1].to_pydatetime()
    window    = pair_settings["Estimation Window"]
    params    = {}
    
    # Fit parameters
    for i in range(window,len(price_data)):
        
        pars = OUParameters()
    
        # Get prices
        asset_a = price_data.iloc[i-window:i,0]
        asset_b = price_data.iloc[i-window:i,1]
        
        # Estimate parameters
        pars.estimate('LN_SPREAD',asset_a,asset_b)
        
        # Save result
        params.update({price_data.index[i]:pars})

    
    allocation = pair_settings['Allocation']
    
    # Create contract objects
    contract_a = Contract(symbols[0])
    contract_b = Contract(symbols[1])
    
    # Create position objects
    PositionA = Position2(contract_a)
    PositionB = Position2(contract_b)
    
    # Create portfolio
    portfolio = Portfolio('Test')
    portfolio.add_position(PositionA)
    portfolio.add_position(PositionB)
    
    start_date   = price_data.index[window].to_pydatetime()
    end_date     = price_data.index[-1].to_pydatetime()
    
    alloc_pars   = AllocationParameters(allocation,
                                        symbols[0],
                                        symbols[1],
                                        horizon_date = None,
                                        risk_tolerance = 0)
    
    
    
    for i in range(window,len(price_data)):
        
        date = price_data.index[i].to_pydatetime()
    
        # Compute ln-spread    
        x = np.log(price_data.iloc[i,0]) - np.log(price_data.iloc[i,1])
        
        # Percentage allocations
        pars     = params[price_data.index[i]]
        tau      = 1.0
        solution = HJB_Spread_Allocator.solve_allocation(alloc_pars,pars,x,tau)
        
        p_a = price_data.iloc[i,0]
        p_b = price_data.iloc[i,1]
    
        
        # Rebalance position in A
        N_a_new = int(solution.alloc_a/(p_a*contract_a.multiplier))
        N_a_old = portfolio.m_positions[symbols[0]].m_position.m_net_position
        delta_N = N_a_new - N_a_old
    
        if(delta_N>0):
            action_a_new = 'BOT'
        elif(delta_N<0):
            action_a_new = 'SLD'
        else:
            action_a_new = None
            
        if(action_a_new!=None):
            
            # Create new trade
            commission = (4.0/10000)*abs(delta_N)*p_a
            
            trade = Trade(symbols[0],action_a_new,p_a,abs(delta_N),commission,date)
            
            # Add the trade to the position
            portfolio.add_trade(trade)
            
            # Update portfolio market value
            portfolio.update_market_value(symbols[0],p_a,p_a,date)        
    
            
        # Rebalance position in B
        N_b_new = int(solution.alloc_b/(p_b*contract_b.multiplier))
        N_b_old = portfolio.m_positions[symbols[1]].m_position.m_net_position
        delta_N = N_b_new - N_b_old
        
        if(delta_N>0):
            action_b_new = 'BOT'
        elif(delta_N<0):
            action_b_new = 'SLD'
        else:
            action_b_new = None
    
        if(action_b_new!=None):
            
            # Create new trade
            commission = (4.0/10000)*abs(delta_N)*p_b
            trade = Trade(symbols[1],action_b_new,p_b,abs(delta_N),commission,date)
            
            # Add the trade to the position
            portfolio.add_trade(trade)
            
            # Update portfolio market value
            portfolio.update_market_value(symbols[1],p_b,p_b,date)     
        
        portfolio.update_market_value(symbols[0],p_a,p_a,date)
        portfolio.update_market_value(symbols[1],p_b,p_b,date)
        
    return portfolio