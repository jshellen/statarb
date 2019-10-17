# -*- coding: utf-8 -*-
#%%

import numpy as np
import dictdiffer
from   copy import deepcopy

from .target import TargetPosition

#%%

class Order:
    """
    Class that encapsulates order related information.
    
    """
    def __init__(self,contract,action,quantity,account):
        
        self.m_contract = contract
        self.m_action   = action
        self.m_quantity = quantity
        self.m_account  = account
        
    @property
    def contract(self):
        return deepcopy(self.m_contract)
    
    @property
    def action(self):
        return self.m_action
    
    @property
    def quantity(self):
        return self.m_quantity

    @property
    def account(self):
        return self.m_account

    def as_dict(self):
        
        return {'TICKER':self.contract.symbol,'ACTION':self.action,
                  'QUANTITY':self.quantity,'ACCOUNT':self.account}

    def __str__(self):
        return f'Order - {self.contract}, Action: {self.action}, Quantity: {self.quantity}, Destination: {self.account}'

class OrderGenerator:
    
    """
    
    Class that encapsulates functions which are used to generate the
    rebalancing orders.
    
    Note: requires package "dictdiffer"
    
    """
    
    def __init__(self,portfolio):
        
        self.m_portfolio = portfolio
        self.m_targets   = {}
        
    @property
    def portfolio(self):
        return deepcopy(self.m_portfolio)
    
    @property
    def targets(self):
        return deepcopy(self.m_targets)
    
    @property
    def current(self):
        c = self.portfolio.positions
        a = {}
        for i,position in c.items():
            a.update({position.contract:position.quantity})
        return a
    
    def generate_target_report(self):
        """
        
        Generates a report showing target positions and the current positions.
        
        """
        a = self.current
        b = self.targets
        # Union of all contracts
        contracts = list(set(list(a.keys()) + list(b.keys())))
        report = {}
        for contract in contracts:
            c_1 = a.get(contract,None)
            c_2 = b.get(contract,None)
            entry = {}
            if(c_1 is not None):
                entry.update({'current':c_1})
            else:
                entry.update({'current':0})
            if(c_2 is not None):
                entry.update({'target':c_2})
            else:
                entry.update({'target':0})
            report.update({contract:entry})
        return report
    
    def update_target_position(self,target,verbose=False):
        """
        Update target position.
        
        """

        if(not isinstance(target,TargetPosition)):
            raise ValueError('Portfolio targets has to be typeof <list>!')
        else:
            self.m_targets.update({target.contract:target.quantity})
            if(verbose):
                print(f'Target for: {target.contract} updated to: {target.quantity}')
    
    
    def get_current_pos_dict(self):
        """
        Returns current position dictionary.
        
        """
        c = self.portfolio.positions
        a = {}
        for i,position in c.items():
                a.update({position.contract:position.quantity})
        return a
    
    
    def on_add_action(self,actions,trades,verbose):
        """
        Generates order information for add actions.
        
        Note: The add actions are given as a list!
        """        
        for act in actions:
            c,q = act
            if(verbose):
                print('Order Generator - Add: ', c,' amount: ' , q)
            if(q==0):
                pass
                #raise ValueError('Order Generator - Quantity cannot be zero!')
            else:
                if(q<0):
                    a = 'S'
                elif(q>0):
                    a = 'B'
                trades.append(Order(c,a,abs(q),self.portfolio.account))
            
    
    def on_remove_action(self,actions,trades,verbose):
        """
        Generates order information for remove actions.
        
        Note: The remove actions are given as a list!
        """
        for act in actions:
            c,q = act
            if(verbose):
                print('Order Generator - Remove: ', c ,' amount: ' , q)
            if(q==0):
                raise ValueError('Order Generator - Quantity cannot be zero!')
            else:
                if(q<0):
                    a = 'B'
                elif(q>0):
                    a = 'S'
                trades.append(Order(c,a,abs(q),self.portfolio.account))    
                
    def on_change_action(self,action,trades,verbose):
        
        c_ls,q = action
        d_q = q[1] - q[0]
        if(verbose):
            print('Order Generator - Change: ', c_ls[0], ' amount: ' , d_q)
        if(d_q==0):
            raise ValueError('Order Generator - Change in quantity cannot be zero!')
        else:
            if(d_q<0):
                a='S'
            elif(d_q>0):
                a='B'
            trades.append(Order(c_ls[0],a,abs(d_q),self.portfolio.account))
    
    def generate_actions(self,verbose=False):
        """
        Generates rebalancing actions.
        
        """        
        if(self.m_targets != {}):
            a = self.current
            b = self.targets
            actions = list(dictdiffer.diff(a, b))
            #print(actions)
            trades = []
            for item in actions:
                if   (item[0] == 'add'   ):
                        self.on_add_action(item[2],trades,verbose)
                elif (item[0] == 'remove'):
                        self.on_remove_action(item[2],trades,verbose)
                elif (item[0] == 'change'):
                        self.on_change_action(item[-2:],trades,verbose)    
                else:
                    raise ValueError(f'Dictdiff result: {item[0]} not understood!')
                
            return trades
        else:
            return None
        
       
        
                
                
                
                
                
                