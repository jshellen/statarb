
import numpy as np

from scipy.linalg import cholesky


def simulate_b(N_sim, N_steps, B_0, mu, sigma_B, dt):
    """
    

    Parameters
    ----------
    N_sim : TYPE
        DESCRIPTION.
    N_steps : TYPE
        DESCRIPTION.
    B_0 : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    sigma_B : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.

    Returns
    -------
    B : TYPE
        DESCRIPTION.

    """
    size = (N_steps, N_sim)
    
    # B(k+1) = B(k) * e^{dM + dW}
    dM = (mu - 0.5 * sigma_B**2) * dt
    dW = sigma_B * np.sqrt(dt) * np.random.normal(0, 1, size)
    B = B_0 * np.exp(np.cumsum(dM + dW, axis=0))

    # Shift and include inception value (t=0).
    B = np.insert(B, 0, B_0, axis=0)
    
    return B    


def simulate_ou_spread(N_sim, N_steps, B_0, X_0, kappa, theta, eta, mu, sigma_B, dt):
    """
    This function simulates Ornstein-Uhlenbeck spread for pairs trading model

    Parameters
    ----------
    N_sim : TYPE
        DESCRIPTION.
    N_steps : TYPE
        DESCRIPTION.
    B_0 : TYPE
        DESCRIPTION.
    X_0 : TYPE
        DESCRIPTION.
    kappa : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.
    eta : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    sigma_B : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.

    Returns
    -------
    A : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    """
    
    size = (N_steps + 1, N_sim)

    # Simulate asset b
    B = simulate_b(N_sim, N_steps, B_0, mu, sigma_B, dt)
    
    # Simulate spread
    X = np.empty(size)
    X[0, :] = X_0
    randn = np.random.normal(0, 1, size)
    for j in range(N_sim):
        for i in range(N_steps):
            dX = kappa*(theta - X[i, j])*dt + eta*np.sqrt(dt) * randn[i, j]
            X[i+1, j] = X[i, j] + dX 
    
    # Simulate price path for A
    A = B * np.exp(X)

    return A, B, X
