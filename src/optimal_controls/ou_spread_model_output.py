# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:33:40 2019

@author: helleju
"""
from copy import deepcopy

from .ou_params import Ornstein_Uhlenbeck_Parameters
from .ou_spread_model_parameters import OU_Spread_Model_Parameters

class OU_Spread_Model_Output:
    
    def __init__(self,opt_alloc,ou_params,model_params,x_ref,tau_ref):
        
        if(not isinstance(opt_alloc,float)):
            raise TypeError('Opt_alloc has to be type of float!')
        
        if(not isinstance(ou_params,Ornstein_Uhlenbeck_Parameters)):
            raise TypeError('OU parameters have to be type of Ornstein_Uhlenbeck_Parameters!')
        
        if(not isinstance(model_params,OU_Spread_Model_Parameters)):
            raise TypeError('Model parameters have to be type of OU_Spread_Model_Parameters!')
        
        if(not isinstance(x_ref,float)):
            raise TypeError('X has to be type of float!')
        
        if(not isinstance(tau_ref,float)):
            raise TypeError('Tau has to be type of float!')
            
        self.m_opt_alloc    = opt_alloc
        self.m_ou_params    = ou_params
        self.m_model_params = model_params
        self.m_x_ref        = x_ref
        self.m_tau_ref      = tau_ref

    @property
    def optimal_allocation(self):
        """
        Returns the optimal allocation.
        
        """
        return self.m_opt_alloc
    
    @property
    def ou_parameters(self):
        """
        Returns a deep copied instance of the Ornstein-Uhlenbec parameters
        used to arrive at the optimal solution.
        """
        return deepcopy(self.m_ou_params)
    
    @property
    def model_parameters(self):
        """
        Returns a deep copied instance of the model parameters used to
        arrive at the optimal solution.
        
        """
        return deepcopy(self.m_model_params)
           
    @property
    def alloc_a(self):
        """
        
        Dollar allocation for asset A given nominal.
        
        TODO: Check for None before multiplying
        
        """
        return self.model_parameters.nominal*self.alloc_a_pct
    
    @property
    def alloc_b(self):
        """
        
        Dollar allocation for asset B given nominal.
        
        TODO: Check for None before multiplying
                
        """        
        return self.model_parameters.nominal*self.alloc_b_pct
    
    @property
    def alloc_a_trunc(self):
        """
        
        Dollar allocation for asset A - truncated to maximum leverage % times
        nominal allocation.
        
        TODO: Check for None before multiplying
                
        """
        return self.model_parameters.nominal*self.alloc_a_pct_trunc

    @property
    def alloc_b_trunc(self):
        """
        
        Dollar allocation for asset B - truncated to maximum leverage % times
        nominal allocation.
        
        TODO: Check for None before multiplying
                
        """
        return self.model_parameters.nominal*self.alloc_b_pct_trunc
    
    @property
    def alloc_a_pct(self):
        """
        
        % allocation for asset A
        
        """        
        return self.m_opt_alloc
    
    @property
    def alloc_b_pct(self):
        """
        
        % allocation for asset B
        
        TODO: Check for None before multiplying
                
        """
        return -self.m_opt_alloc
    
    @property
    def alloc_a_pct_trunc(self):
        """
        
        % allocation for asset A - truncated to maximum leverage %
        
        TODO: Check for None before operations
                
        """            
        if(self.m_opt_alloc<0):
            pct_a = max(-self.model_parameters.maximum_leverage,self.m_opt_alloc)
        else:
            pct_a = min(self.model_parameters.maximum_leverage,self.m_opt_alloc)
        return pct_a
        
    
    @property
    def alloc_b_pct_trunc(self):
        """
        
        % allocation for asset B - truncated to maximum leverage %
        
        TODO: Check for None before operations
        
        """                
        if(self.opt_alloc<0):
            pct_b = min(self.model_parameters.maximum_leverage,-self.opt_alloc)
        else:
            pct_b = max(-self.model_parameters.maximum_leverage,-self.opt_alloc)
        return pct_b   