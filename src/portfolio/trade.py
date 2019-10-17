# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:34:09 2019

@author: helleju
"""

class Trade:
    """
    Convenience class to keep records of a trade.
    """    
    def __init__(self,symbol,action,price,quantity,commission,time):
        
        if(not isinstance(symbol,str)):
            raise ValueError('Symbol needs to be <str>!')
            
        if(not isinstance(action,str)):
            raise ValueError('Action needs to be <str>!')
        else:
            if(action not in ['BOT','SLD']):
                raise ValueError('Action needs to be either "BOT" or "SLD"!')
                
        if(not isinstance(price,(float,int))):
            raise ValueError('Price needs to be either <float> or <int>!')
        
        if(not isinstance(quantity,int)):
            raise ValueError('Quantity needs to be <int>!')
        
        if(not isinstance(commission,(float,int))):
            raise ValueError('Commission needs to be either <float> or <int>!')
        
        self.m_symbol     = symbol
        
        self.m_action     = action
        
        self.m_price      = price
        
        self.m_quantity   = quantity
        
        self.m_commission = commission
        
        self.m_time       = time
        
    @property
    def symbol(self):
        return self.m_symbol 
    
    @property
    def action(self):
        return self.m_action
    
    @property
    def price(self):   
        return self.m_price
        
    @property
    def quantity(self):  
        return self.m_quantity

    @property
    def time(self):  
        return self.m_time
        
    @property
    def commission(self):  
        return self.m_commission

