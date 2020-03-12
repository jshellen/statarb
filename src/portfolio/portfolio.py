# -*- coding: utf-8 -*-

from .position import Position2
import pandas


class Portfolio:
    """
    Convenience class for storing individual positions and for generating
    various reports.
    """
    def __init__(self, name):
        
        self.m_name = name
        self.m_positions = {}

    @property
    def name(self):
        return self.m_name

    def add_position(self, position):
        """
        Add a new position to portfolio. Raises value error if the input position
        has a symbol that already exist in the portfolio.
        """
        if position.contract.symbol in self.m_positions:
            raise ValueError('Position already exists for symbol: {position.contract.symbol}!')
        else:
            self.m_positions.update({position.contract.symbol: position})
    
    def add_trade(self, trade):
        """
        Add a new trade to a position in the portfolio.
        """
        if trade.symbol in self.m_positions:
            self.m_positions[trade.symbol].add_trade(trade)
        else: 
            raise ValueError('Cannot add trade. No position exists for symbol: {trade.symbol}!')

    def get_position(self, symbol):
        """
        Get current position in a symbol.
        """
        if symbol in self.m_positions:
            return self.m_positions[symbol]
        else:
            print(f'Position with symbol: {symbol} not found from portfolio.')
            return None

    def update_market_value(self, symbol, bid, ask, time=None):
        """
        Update market value of a position given symbol and corresponding
        bid and ask prices.
        """
        if symbol in self.m_positions:
            self.m_positions[symbol].update_market_value(bid, ask, time)
        else: 
            raise ValueError('Cannot update market value. No position exists for symbol: {trade.symbol}!')        
    
