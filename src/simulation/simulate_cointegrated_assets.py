
import numpy as np

def simulate_b(N_sim, N_steps,B_0,mu,sigma_B,dt):
    """
    
    Simulate asset B
    
    """
    size = (N_steps,N_sim)
    #dt   = 1.0/N_steps
    
    # Simulate price path for B
    B    = np.zeros(N_sim)
    B[:] = B_0
    
    # Diffusion
    dW = np.exp(sigma_B*np.sqrt(dt)*np.random.normal(0,1,size))
    
    # Drift
    dM = np.exp((mu-0.5*sigma_B**2)*dt)
    
    B  = B[:]*np.cumprod(dM*dW,axis=0)
    
    return B    


def simulate_cointegrated_assets(N_sim,N_steps,B_0,mu,kappa,theta,eta,sigma_B,dt):
    
    size = (N_steps,N_sim)
    #dt   = 1.0/N_steps
    
    # Simulate price path for B
    B = simulate_b(N_sim,N_steps,B_0,mu,sigma_B,dt)
    
    #Simulate spread path using Euler-Maruyama method
    X    = np.zeros(size)
    X[:] = theta
    for j in range(0,N_sim):
        for i in range(0,N_steps-1):
            dX = kappa*(theta - X[i,j])*dt + eta*np.sqrt(dt)*np.random.normal(0,1)
            X[i+1,j]    = X[i,j] + dX
    
    # Simulate price path for A
    A = B*np.exp(X)

    return A,B,X