# -*- coding: utf-8 -*-


class Contract:
    
    def __init__(self, symbol, security_type, multiplier=1):
        
        self.m_symbol = symbol
        self.m_sec_type = security_type
        self.m_multiplier = multiplier
    
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

        return f"Symbol: {self.symbol} Sec. Type: {self.sec_type} Multiplier: {self.multiplier}"
    
    def __eq__(self, other):
        """
        Equal operator overload.
        
        Here comparison is made based on the
        symbol, security type (sec_type) and price point change multiplier
        (multiplier).
        """
        if( (self.symbol == other.symbol)
        and (self.sec_type == other.sec_type)
        and (self.multiplier == self.multiplier)):
            return True
        else:
            return False
        
