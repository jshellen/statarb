# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:32:18 2019

@author: helleju
"""
#%%
from .position_info import PositionInfo
import pandas
#%%
class Portfolio:
    """
    Convenience class for storing individual positions and for generating
    various reports.
    """
    def __init__(self,name):
        
        self.m_name      = name
        self.m_positions = {}
    
    
    @property
    def name(self):
        return self.m_name
    
    
    
    def add_position(self,position):
        """
        Add a new position to portfolio. Raises value error if the input position
        has a symbol that already exist in the portfolio.
        """
        if(position.contract.symbol in self.m_positions):
            raise ValueError('Position already exists for symbol: {position.contract.symbol}!')
        else:
            position_info = PositionInfo(position)
            self.m_positions.update({position.contract.symbol: position_info})
    
    def add_trade(self,trade):
        """
        Add a new trade to a position in the portfolio.
        """
        if(trade.symbol in self.m_positions):
            self.m_positions[trade.symbol].add_trade(trade)
        else: 
            raise ValueError('Cannot add trade. No position exists for symbol: {trade.symbol}!')
    
    def update_market_value(self,symbol,bid,ask,time):
        """
        
        """
        if(symbol in self.m_positions):
            self.m_positions[symbol].m_position.update_market_value(bid,ask)
            self.m_positions[symbol].log_position_status(time)
        else: 
            raise ValueError('Cannot update market value. No position exists for symbol: {trade.symbol}!')        
    
    def generate_holding_report(self,formate='frame'):
        
        report = {}
        for symbol,position_info in self.m_positions.items():
            report.update({symbol:position_info.m_position.m_net_position})
        
        if(formate=='frame'):
            report = pandas.DataFrame.from_dict(report,orient='index')
        
        return report
        
    def generate_position_pnl_report(self,symbol,formate='frame'):
        """
        Generate Profit and Loss report for a single symbol in the portfolio
        """
        if(symbol in self.m_positions):
            report = self.m_positions[symbol].generate_pnl_report(formate)
        else: 
            raise ValueError('Cannot generate PNL report. No position exists for symbol: {trade.symbol}!')   
        return report
    
    def generate_position_pnl_reports(self,formate='frame'):
        """
        Generate Profit and Loss report for all symbols in the portfolio.
        """        
        reports = {}
        for symbol,position in self.m_positions.items():
            report = position.generate_pnl_report(formate)
            reports.update({symbol:report})
        return reports
    
    def generate_portfolio_pnl_report(self):
        """
        Generate Profit and Loss report for the entire portfolio.
        """
        reports = self.generate_position_pnl_reports(formate='frame')
        pnls = [] 
        for symbol,report in reports.items():
            pnl = report['TOTAL_PNL'].to_frame()
            pnl.columns = [symbol]
            pnls.append(pnl)
        pnls = pandas.concat(pnls,axis=1)
        pnls['TOTAL_PNL'] = pnls.sum(axis=1)
        return pnls
    