
import numpy as np

def geometrix_brownian_motion(S0, mu, sigma, dt, N_steps, N_paths):
    
    size = (N_steps, N_paths)
    s = np.sqrt(dt) * np.random.normal(0, sigma, size) + mu * dt * np.ones(size)
    
    return np.cumprod(np.exp(s), axis=0)
