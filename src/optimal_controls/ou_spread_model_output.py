# -*- coding: utf-8 -*-

from copy import deepcopy

from .ou_params import OrnsteinUhlenbeckProcessParameters
from .ou_spread_model_parameters import OUSpreadModelStrategyParameters


class OUSpreadModelOutput:
    
    def __init__(self, opt_alloc, model_params, strategy_params, x_ref, tau_ref):
        
        if not isinstance(opt_alloc, (int, float)):
            raise TypeError('Opt_alloc has to be <int> or <float>')
        
        if not isinstance(model_params, OrnsteinUhlenbeckProcessParameters):
            raise TypeError('OU parameters have to be <OUSpreadModelStrategyParameters>.')
        
        if not isinstance(strategy_params, OUSpreadModelStrategyParameters):
            raise TypeError('Model parameters have to be type of OU_Spread_Model_Parameters!')
        
        if not isinstance(x_ref, (int, float)):
            raise TypeError('X has to be <int> or <float>')
        
        if not isinstance(tau_ref, (int, float)):
            raise TypeError('Tau has to be <int> or <float>.')
            
        self.m_opt_alloc = opt_alloc
        self.m_model_params = model_params
        self.m_strategy_params = strategy_params
        self.m_x_ref = x_ref
        self.m_tau_ref = tau_ref

    @property
    def optimal_allocation(self):
        """
        Returns the optimal allocation.
        
        """
        return self.m_opt_alloc
    
    @property
    def model_parameters(self):
        """
        Returns a deep copied instance of the Ornstein-Uhlenbec parameters
        used to arrive at the optimal solution.
        """
        return deepcopy(self.m_model_params)
    
    @property
    def strategy_parameters(self):
        """
        Returns a deep copied instance of the model parameters used to
        arrive at the optimal solution.
        
        """
        return deepcopy(self.m_strategy_params)
           
    @property
    def alloc_a(self):
        """
        
        Dollar allocation for asset A given nominal.
        
        TODO: Check for None before multiplying
        
        """
        return self.strategy_parameters.nominal * self.alloc_a_pct
    
    @property
    def alloc_b(self):
        """
        
        Dollar allocation for asset B given nominal.
        
        TODO: Check for None before multiplying
                
        """        
        return self.strategy_parameters.nominal * self.alloc_b_pct
    
    @property
    def alloc_a_trunc(self):
        """
        
        Dollar allocation for asset A - truncated to maximum leverage % times
        nominal allocation.
        
        TODO: Check for None before multiplying
                
        """
        return self.strategy_parameters.nominal * self.alloc_a_pct_trunc

    @property
    def alloc_b_trunc(self):
        """
        
        Dollar allocation for asset B - truncated to maximum leverage % times
        nominal allocation.
        
        TODO: Check for None before multiplying
                
        """
        return self.strategy_parameters.nominal * self.alloc_b_pct_trunc
    
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
        if self.m_opt_alloc < 0:
            pct_a = max(-self.strategy_parameters.maximum_leverage, self.m_opt_alloc)
        else:
            pct_a = min(self.strategy_parameters.maximum_leverage, self.m_opt_alloc)

        return pct_a
        
    
    @property
    def alloc_b_pct_trunc(self):
        """
        
        % allocation for asset B - truncated to maximum leverage %
        
        TODO: Check for None before operations
        
        """                
        if self.m_opt_alloc < 0:
            pct_b = min(self.strategy_parameters.maximum_leverage, -self.m_opt_alloc)
        else:
            pct_b = max(-self.strategy_parameters.maximum_leverage, -self.m_opt_alloc)

        return pct_b
