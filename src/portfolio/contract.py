# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:34:20 2019

@author: helleju
"""
multipliers = {'HG':250,'PL':50}

def get_multiplier(ticker):    
    
    out = None
    if   (ticker.find("Equity")>0):
        out = 1
    elif (ticker.find("Comdty")>0):
        symbol  = ticker[0:2].replace(" ","")
        if(symbol not in multipliers):
            raise ValueError(f'Symbol: {symbol} not found from multiplier dictionary!')
        out = multipliers.get(symbol)
    else:
        raise ValueError(f'Ticker: {ticker} not understood!')
    return out

def get_instrument_type(ticker):
    
    out = None
    if   (ticker.find("Equity")>0):
        out = 'STK'
    elif (ticker.find("Comdty")>0):
        out = 'F'
    elif (ticker.find("Index")>0):
        out = 'F'
    else:
        raise ValueError(f'Ticker: {ticker} not understood!')
    return out
        

class Contract:
    
    def __init__(self,symbol):
        
        self.m_symbol     = symbol
        self.m_sec_type   = get_instrument_type(symbol)
        self.m_multiplier = get_multiplier(symbol)
    
    @property
    def symbol(self):
        return self.m_symbol

    @property
    def sec_type(self):
        return self.m_sec_type
    
    @property
    def multiplier(self):
        return self.m_multiplier
    
    
    
    
    
    def __str__(self):
        """
        Output operator overload.
        """
        return f"Symbol: {self.symbol} Sec. Type: {self.sec_type} Multiplier: {self.multiplier}"
    
    def __eq__(self,other):
        """
        Equal operator overload.
        
        Here comparison is made based on the
        symbol, security type (sec_type) and price point change multiplier
        (multiplier).
        """
        if( (self.symbol     == other.symbol) 
        and (self.sec_type   == other.sec_type)
        and (self.multiplier == self.multiplier)):
            return True
        else:
            return False
        
    def __hash__(self):
        """
        Hash operator overload.
        
        This is needed in order to use dictdiffer when computing rebalancing
        orders.
        """
        return hash(str(self))
        
        
        