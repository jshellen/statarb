# -*- coding: utf-8 -*-

from pandas.core.frame  import DataFrame
from pandas.core.series import Series
from pandas             import concat
from scipy.stats.stats  import pearsonr 
from numpy              import log, sqrt, ndarray

from .estimate_ou_params import estimate_ou_parameters

NONE_TYPE = type(None)

class Ornstein_Uhlenbeck_Parameters:
    """
    Encapsulates Ornstein Uhlenbeck process parameters.
    
    """
    def __init__(self,kappa=None,theta=None,eta=None,sigma_B=None,rho=None):
        
        if(not isinstance(kappa,NONE_TYPE)):
            if(not isinstance(kappa,float)):
                raise TypeError(f'Kappa has to be float instead of {type(kappa)}!')
                
        if(not isinstance(theta,NONE_TYPE)):
            if(not isinstance(theta,float)):
                raise TypeError('theta has to be float!')
                
        if(not isinstance(eta,NONE_TYPE)):
            if(not isinstance(eta,float)):
                raise TypeError('eta has to be float!')
                
        if(not isinstance(sigma_B,NONE_TYPE)):
            if(not isinstance(sigma_B,float)):
                raise TypeError('sigma_B has to be float!')
                    
        if(not isinstance(rho,NONE_TYPE)):
            if(not isinstance(rho,float)):
                raise TypeError('rho has to be float!')
                
        # Initialize values to none
        self.m_eta     = eta
        self.m_sigma_B = sigma_B
        self.m_rho     = rho
        self.m_theta   = theta
        self.m_kappa   = kappa
    
    
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
        series using ordinary least squares estimation.
        
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
        
        if(isinstance(self.kappa,float)):
            kappa = round(self.kappa,2)
        else:
            kappa = ''
        
        if(isinstance(self.theta,float)):
            theta = round(self.theta,2)
        else:
            theta = ''
        
        if(isinstance(self.rho,float)):
            rho = round(self.rho,2)
        else:
            rho = ''
        
        if(isinstance(self.eta,float)):
            eta = round(self.eta,2)
        else:
            eta = ''
        
        return f"Ornstein-Uhlenbeck Parameters: Kappa = {kappa}, Theta = {theta}, Rho = {rho}, Eta = {eta}"
