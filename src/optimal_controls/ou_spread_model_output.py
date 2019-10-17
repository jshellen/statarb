# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:33:40 2019

@author: helleju
"""
from copy import deepcopy

class OU_Spread_Model_Output:
    
    """
    
       Encapsulates optimal allocation in pairs trading portfolio as solved by
       Supakorn Mudchanatongsuk, James A. Primbs and Wilfred Wong
       
       http://folk.ntnu.no/skoge/prost/proceedings/acc08/data/papers/0479.pdf
       
    """
    
    def __init__(self,opt_alloc,model_params,x_ref,tau_ref):
        
        self.m_opt_alloc    = opt_alloc
        self.m_model_params = model_params
        self.m_x_ref        = x_ref
        self.m_tau_ref      = tau_ref

    @property
    def optimal_allocation(self):
        """
        Returns a deep copied instance of the optimal allocation.
        
        """
        return deepcopy(self.m_opt_alloc)
    
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
        
        """
        return self.model_parameters.nominal*self.alloc_a_pct
    
    @property
    def alloc_b(self):
        """
        
        Dollar allocation for asset B given nominal.
        
        """        
        return self.model_parameters.nominal*self.alloc_b_pct
    
    @property
    def alloc_a_trunc(self):
        """
        
        Dollar allocation for asset A - truncated to maximum leverage % times
        nominal allocation.
        
        """
        return self.model_parameters.nominal*self.alloc_a_pct_trunc

    @property
    def alloc_b_trunc(self):
        """
        
        Dollar allocation for asset B - truncated to maximum leverage % times
        nominal allocation.
        
        """
        return self.model_parameters.nominal*self.alloc_b_pct_trunc
    
    @property
    def alloc_a_pct(self):
        """
        
        % allocation for asset A
        
        """        
        return self.opt_alloc
    
    @property
    def alloc_b_pct(self):
        """
        
        % allocation for asset B
        
        """
        return -self.opt_alloc
    
    @property
    def alloc_a_pct_trunc(self):
        """
        
        % allocation for asset A - truncated to maximum leverage %
        
        """            
        if(self.opt_alloc<0):
            pct_a = max(-self.model_parameters.maximum_leverage,self.opt_alloc)
        else:
            pct_a = min(self.model_parameters.maximum_leverage,self.opt_alloc)
        return pct_a
        
    
    @property
    def alloc_b_pct_trunc(self):
        """
        
        % allocation for asset B - truncated to maximum leverage %
        
        """                
        if(self.opt_alloc<0):
            pct_b = min(self.model_parameters.maximum_leverage,-self.opt_alloc)
        else:
            pct_b = max(-self.model_parameters.maximum_leverage,-self.opt_alloc)
        return pct_b   