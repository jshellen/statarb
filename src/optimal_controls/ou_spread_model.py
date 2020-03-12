# -*- coding: utf-8 -*-

from numpy import sqrt, exp, finfo
from .ou_params import OrnsteinUhlenbeckProcessParameters
from .ou_spread_model_parameters import OUSpreadModelStrategyParameters
from .ou_spread_model_output import OUSpreadModelOutput


class OUSpreadModelSolver:
    
    @staticmethod
    def solve_alpha(gamma, kappa, eta, tau):

        if not isinstance(gamma, (float, int)):
            raise TypeError('Gamma needs to be type of float')
            
        if not isinstance(kappa, (float, int)):
            raise TypeError('Kappa needs to be type of float')        

        if not isinstance(eta, (float, int)):
            raise TypeError('Eta needs to be type of float') 
            
        if not isinstance(tau, (float, int)):
            raise TypeError('Tau needs to be type of float') 
   
        if gamma > 1.0:
            raise ValueError('Gamma has to be lower than 1.')
            
        if tau < 0.0:
            raise ValueError('Tau cannot be negative.')
            
        a = sqrt(1.0-gamma)
        
        if abs(a) < 10e-16:
            raise ValueError('gamma too close to one. Will result in zero division in alpha function.')
        else:
            t_1 = (kappa*(1-a))/(2*eta**2)
            t_2 = 1.0+(2*a)/(1-a-(1+a)*exp((2*kappa*tau)/a))
            
        return t_1*t_2
    
    @staticmethod
    def solve_beta(gamma, kappa, theta, eta, rho, sigma, tau):

        if not isinstance(gamma, (float, int)):
            raise TypeError('Gamma needs to be type of float')
            
        if not isinstance(kappa, (float, int)):
            raise TypeError('Kappa needs to be type of float')        

        if not isinstance(theta, (float, int)):
            raise TypeError('Theta needs to be type of float')           

        if not isinstance(eta, (float, int)):
            raise TypeError('Eta needs to be type of float') 
            
        if not isinstance(rho, (float, int)):
            raise TypeError('Rho needs to be type of float') 

        if not isinstance(tau, (float, int)):
            raise TypeError('Tau needs to be type of float') 
   
        if gamma >= 1.0:
            raise ValueError('Gamma has to be strictly lower than 1.0!')
            
        if tau < 0.0:
            raise ValueError('Tau cannot be negative.')
            
        a = sqrt(1.0-gamma)

        # Machine epsilon to prevent zero division
        eps = finfo(float).eps
        
        b = exp(2*kappa*tau/(a+eps))
        t_1 = 1.0/((2*eta**2)*((1-a)-(1+a)*exp((2*kappa*tau)/(a+eps))))
        t_2 = gamma*a*(eta**2 + 2*rho*sigma*eta)*((1-b)**2)
        t_3 = -gamma*(eta**2 + 2*rho*sigma*eta + 2*kappa*theta)*(1-b)
            
        return t_1*(t_2+t_3)

    @staticmethod
    def solve_h_prime(gamma, kappa, theta, eta, sigma_b, rho, tau, x):

        if not isinstance(gamma, (float, int)):
            raise TypeError('Gamma needs to be type of float!')
            
        if not isinstance(kappa, (float, int)):
            raise TypeError('Kappa needs to be type of float!')        

        if not isinstance(theta, (float, int)):
            raise TypeError('Theta needs to be type of float!')           

        if not isinstance(eta, (float, int)):
            raise TypeError('Eta needs to be type of float!') 

        if not isinstance(sigma_b, (float, int)):
            raise TypeError('sigma_b needs to be type of float!')
            
        if not isinstance(rho, (float, int)):
            raise TypeError('Rho needs to be type of float!') 

        if not isinstance(tau, (float, int)):
            raise TypeError('Tau needs to be type of float!') 
   
        if not isinstance(x, (float, int)):
            raise TypeError('X needs to be type of float!') 
            
        if gamma > 1.0:
            raise ValueError('Gamma has to be strictly lower than 1!')
            
        if tau < 0.0:
            raise ValueError('Tau cannot be negative!')
        
        if eta < 0.0:
            raise ValueError('Eta cannot be negative!')

        if sigma_b < 0.0:
            raise ValueError('Sigma_B cannot be negative!')
        
        # Solve alpha
        a = OUSpreadModelSolver.solve_alpha(gamma, kappa, eta, tau)
        
        # Solve beta
        b = OUSpreadModelSolver.solve_beta(gamma, kappa, theta, eta, rho, sigma_b, tau)
        
        # Solve optimal solution "h"
        
        # Machine epsilon to prevent division by zero
        eps = finfo(float).eps
        
        h = (1.0/(1.0-gamma + eps))*(b + 2*x*a - (kappa*(x-theta))/(eta**2 + eps) + (rho*sigma_b)/(eta+eps) + 0.5)
        
        return h
    
    @staticmethod
    def solve_asset_weights(model_params, strategy_params, spread_level, time_left):
        """

        """
        if not isinstance(model_params, OrnsteinUhlenbeckProcessParameters):
            raise TypeError('OU parameters have to be type of <OrnsteinUhlenbeckProcessParameters>.')
        
        if not isinstance(strategy_params, OUSpreadModelStrategyParameters):
            raise TypeError('Model parameters have to be type of <OU_Spread_Model_Parameters>.')
        
        if not isinstance(spread_level, (float, int)):
            raise TypeError('X has to be type of float!')
        
        if not isinstance(time_left, (float, int)):
            raise TypeError('Tau has to be type of float!')
            
        solution = OUSpreadModelSolver.solve_h_prime(strategy_params.risk_tolerance, model_params.kappa,
                                                     model_params.theta, model_params.eta,
                                                     model_params.sigma_b, model_params.rho, time_left, spread_level)

        out = OUSpreadModelOutput(solution, model_params, strategy_params, spread_level, time_left)
        
        return out

 


