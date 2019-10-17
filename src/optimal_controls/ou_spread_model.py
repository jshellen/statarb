# -*- coding: utf-8 -*-

from numpy import sqrt, exp

from .ou_spread_model_parameters import OU_Spread_Model_Parameters

from .ou_spread_model_output import OU_Spread_Model_Output


class OU_Spread_Model:
    
    
    @staticmethod
    def solve_alpha(gamma,kappa,eta,tau):
        """
        Equation ??.?? from page ??
        
        """
        if(not isinstance(gamma,float)):
            raise TypeError('Gamma needs to be type of float')
            
        if(not isinstance(kappa,float)):
            raise TypeError('Kappa needs to be type of float')        

        if(not isinstance(eta,float)):
            raise TypeError('Eta needs to be type of float') 
            
        if(not isinstance(tau,float)):
            raise TypeError('Tau needs to be type of float') 
   
        if(gamma>1.0):
            raise ValueError('Gamma has to be lower than 1.')
            
        if(tau<0.0):
            raise ValueError('Tau cannot be negative.')
            
        a = sqrt(1.0-gamma)
        
        if(abs(a)<10e-16):
            raise ValueError('Encountered zero division in alpha function')
        else:
            t_1 = (kappa*(1-a))/(2*eta*
                  *2)
            t_2 = 1.0+(2*a)/(1-a-(1+a)*exp((2*kappa*(tau))/a))
            
        return t_1*t_2
    
    @staticmethod
    def solve_beta(gamma,kappa,theta,eta,rho,sigma,tau):
        """
        Equation ??.?? from page ??
        
        """
        if(not isinstance(gamma,float)):
            raise TypeError('Gamma needs to be type of float')
            
        if(not isinstance(kappa,float)):
            raise TypeError('Kappa needs to be type of float')        

        if(not isinstance(theta,float)):
            raise TypeError('Theta needs to be type of float')           

        if(not isinstance(eta,float)):
            raise TypeError('Eta needs to be type of float') 
            
        if(not isinstance(rho,float)):
            raise TypeError('Rho needs to be type of float') 

        if(not isinstance(tau,float)):
            raise TypeError('Tau needs to be type of float') 
   
        if(gamma>=1.0):
            raise ValueError('Gamma has to be strictly lower than 1.0!')
            
        if(tau<0.0):
            raise ValueError('Tau cannot be negative!')
            
        a = sqrt(1.0-gamma)
        
        #if(abs(a)<10e-16):
        #    raise ValueError('Encountered zero division in beta function')
        #else:
        
        # Machine epsilon to prevent zero division
        eps = np.finfo(float).eps
        
        b   = exp(2*kappa*(tau)/ (a+eps) )
        t_1 = 1.0/((2*eta**2)*((1-a)-(1+a)*exp((2*kappa*(tau))/ (a+eps)   )))
        t_2 = gamma*a*(eta**2 + 2*rho*sigma*eta)*((1-b)**2)
        t_3 = -gamma*(eta**2 + 2*rho*sigma*eta + 2*kappa*theta)*(1-b)
            
        return t_1*(t_2+t_3)
        
     
    @staticmethod
    def solve_h_prime(gamma,kappa,theta,eta,sigma_B,rho,tau,x):
        """
        Equation ??.?? from page ??
        
        """
        if(not isinstance(gamma,float)):
            raise TypeError('Gamma needs to be type of float!')
            
        if(not isinstance(kappa,float)):
            raise TypeError('Kappa needs to be type of float!')        

        if(not isinstance(theta,float)):
            raise TypeError('Theta needs to be type of float!')           

        if(not isinstance(eta,float)):
            raise TypeError('Eta needs to be type of float!') 

        if(not isinstance(sigma_B,float)):
            raise TypeError('sigma_B needs to be type of float!') 
            
        if(not isinstance(rho,float)):
            raise TypeError('Rho needs to be type of float!') 

        if(not isinstance(tau,float)):
            raise TypeError('Tau needs to be type of float!') 
   
        if(not isinstance(x,float)):
            raise TypeError('X needs to be type of float!') 
            
        if(gamma>1.0):
            raise ValueError('Gamma has to be strictly lower than 1!')
            
        if(tau<0.0):
            raise ValueError('Tau cannot be negative!')
        
        if(eta<0.0):
            raise ValueError('Eta cannot be negative!')

        if(sigma_B<0.0):
            raise ValueError('Sigma_B cannot be negative!')
        
        # Solve alpha
        a = OU_Spread_Model.solve_alpha(gamma,kappa,eta,tau)
        
        # Solve beta
        b = OU_Spread_Model.solve_beta(gamma,kappa,theta,eta,rho,sigma_B,tau)
        
        # Solve optimal solution "h"
        
        # Machine epsilon to prevent division by zero
        eps = np.finfo(float).eps
        
        h = (1.0/(1.0-gamma + eps))*(b + 2*x*a - (kappa*(x-theta))/(eta**2 + eps) + (rho*sigma_B)/(eta+eps) + 0.5)    
        
        return h
    
    @staticmethod
    def solve_allocation(model_params,x,tau):
        """
        
        Optimal allocation to asset A given:
            
            1) Model parameters
    
            2) Current spread level
    
            3) Trading time remaining
        
        """
        
        if(not isinstance(model_params,OU_Spread_Model_Parameters)):
            raise TypeError('Model parameters have to be type of OU_Spread_Model_Parameters!')
        
        if(not isinstance(x,float)):
            raise TypeError('X has to be type of float!')
        
        if(not isinstance(tau,float)):
            raise TypeError('Tau has to be type of float!')
            
        
        # Compute optimal trading allocation percentage
        gamma = model_params.risk_tolerance
        
        if(gamma>=1):
            raise ValueError('Gamma has to be < 1. Otherwise we get imaginary solution.')
        else:
        
            kappa   = model_params.kappa
            eta     = model_params.eta
            theta   = model_params.theta
            sigma_B = model_params.sigma_B
            rho     = model_params.rho
            
            solution = OU_Spread_Model.solve_h_prime(gamma,kappa,theta,eta,sigma_B,rho,tau,x)

        out = OU_Spread_Model_Output(solution,model_params,model_params,x,tau)
        
        return out

 


