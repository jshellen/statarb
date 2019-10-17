# -*- coding: utf-8 -*-

from pandas.core.frame  import DataFrame
from pandas.core.series import Series
from pandas             import concat
from scipy.stats.stats  import pearsonr 
from numpy              import log, sqrt, ndarray

from .estimate_ou_params import estimate_ou_parameters

class Ornstein_Uhlenbeck_Parameters:
    """
    Encapsulates Ornstein Uhlenbeck process parameters.
    
    """
    def __init__(self,json=None):
        
        
        if(json is not None):
            if(not isinstance(json,dict)):
                raise TypeError('Input has to be a dictionary!')
        
            if("eta" not in json):
                raise ValueError('Eta was not defined in parameters!')
            else:
                if(not isinstance(json["eta"],float)):
                    raise TypeError('Eta has to be float!')
                    
                self.eta = json["eta"]
                
            if("sigma_B" not in json):
                raise ValueError('sigma_B was not defined in parameters!')
            else:
                if(not isinstance(json["sigma_B"],float)):
                    raise TypeError('sigma_B has to be float!')
                    
                self.sigma_B = json["sigma_B"]   
                
            if("rho" not in json):
                raise ValueError('rho was not defined in parameters!')
            else:
                if(not isinstance(json["rho"],float)):
                self.rho = json["rho"]     
            if("theta" not in json):
                raise ValueError('theta was not defined!')
            else:
                self.theta = json["theta"]    
            if("kappa" not in json):
                raise ValueError('kappa was not defined!')
            else:
                self.kappa = json["kappa"]  
        else:                 
            
            # Initialize values to none
            self.m_eta     = None
            self.m_sigma_B = None
            self.m_rho     = None
            self.m_theta   = None
            self.m_kappa   = None
    
    
    @property
    def kappa(self):
        """
        Mean-reversion speed of the process.
        """
        return self.m_kappa
    
    @property
    def theta(self):
        """
        Long-term average level of the spread
        """
        return self.m_theta
        
    @property
    def rho(self):
        """
        Correlation coefficient between the spread and the price series
        """
        return self.m_rho

    @property
    def eta(self):
        """
        Volatility of the spread.
        """
        return self.m_eta
    
    @property
    def sigma_B(self):
        """
        Volatility of asset "B"
        """
        return self.m_sigma_B
    
    
    def estimate_using_ols(self,A_data,B_data,dt):
        """
        Estimates Ornstein Uhlenbeck stochastic process parameters from price
        series.
        
        If the estimation is successfull the method returns 1, else 0.
        
        """
        
        if(not isinstance(A_data,(DataFrame,Series,ndarray))):
            raise TypeError('Price data A needs to be type of pandas.core.frame.DataFrame, pandas.core.series.Series or numpy.ndarray')

        if(not isinstance(B_data,(DataFrame,Series,ndarray))):
            raise TypeError('Price data B needs to be type of pandas.core.frame.DataFrame, pandas.core.series.Series or numpy.ndarray')
        
        if(not isinstance(dt,float)):
            raise TypeError('Delta time has to be type of float!')
        
        if(dt<=0):
            raise ValueError('Delta time has to be positive and non-zero!')
        
        if(isinstance(A_data,Series)):
            A_data = A_data.to_frame(name=0)
            
        if(isinstance(B_data,Series)):
            B_data = B_data.to_frame(name=0)

        if(isinstance(A_data,ndarray)):
            A_data = DataFrame(data = A_data)
            
        if(isinstance(B_data,ndarray)):
            B_data = DataFrame(data = B_data)
        
        try:
                
            # Compute spread
            spread = log(A_data) - log(B_data)
            
            # Estimate OU parameters
            pars = estimate_ou_parameters(spread.values,dt)
            
            self.m_kappa = pars[0]
            self.m_theta = pars[1]
            self.m_eta   = pars[2]        
            
            a = B_data.pct_change(1)
            b = spread.diff(1)
            c = concat([a,b],axis=1).dropna()
            self.m_rho = pearsonr(c.iloc[:,0],c.iloc[:,1])[0]
            self.m_sigma_B = a.std().values[0]*sqrt(1.0/dt)
        
            return 1
        
        except Exception as e:
            
            print(e)
            
            return 0
    
    def __str__(self):
        
        if(self.kappa is not None):
            kappa = round(self.kappa,2)
        else:
            kappa = self.kappa
        
        if(self.theta is not None):
            theta = round(self.theta,2)
        else:
            theta = self.theta
        
        if(self.rho is not None):
            rho   = round(self.rho,2)
        else:
            rho   = self.rho
        
        if(self.eta is not None):
            eta   = round(self.eta,2)
        else:
            eta   = self.eta
        
        return f"Ornstein-Uhlenbeck Parameters: Kappa = {kappa}, Theta = {theta}, Rho = {rho}, Eta = {eta}"
