# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:48:57 2019

@author: helleju
"""

from copy import deepcopy

class TargetPosition:

    def __init__(self,contract,quantity):
        
        self.m_contract = contract
        self.m_quantity = quantity
        
    @property
    def contract(self):
        return deepcopy(self.m_contract)
    
    @property
    def quantity(self):
        return self.m_quantity
    
