
import numpy as np

from scipy.linalg import cholesky


def simulate_b(N_sim, N_steps, B_0, mu, sigma_B, dt):
    """
    
    Simulate logNormal asset B
    
    """
    size = (N_steps, N_sim)
    
    # B(k+1) = B(k) * e^{dM + dW}
    dM = (mu - 0.5 * sigma_B**2) * dt
    dW = sigma_B * np.sqrt(dt) * np.random.normal(0, 1, size)
    B = B_0 * np.exp(np.cumsum(dM + dW, axis=0))

    # Shift and include inception value (t=0).
    B = np.insert(B, 0, B_0, axis=0)
    
    return B    


def simulate_ou_spread(N_sim, N_steps, B_0, mu, kappa, theta, eta, sigma_B, dt):
    
    size = (N_steps + 1, N_sim)
    #np.random.seed(0)

    B = simulate_b(N_sim, N_steps, B_0, mu, sigma_B, dt)

    X = np.full(size, fill_value=theta)
    randn = np.random.normal(0, 1, size)
    for j in range(N_sim):
        for i in range(N_steps):
            dX = kappa*(theta - X[i, j])*dt + eta*np.sqrt(dt) * randn[i, j]
            X[i+1, j] = X[i, j] + dX #TODO: consider having simulation in 1st dim, time in 2nd, for faster looping (C-contiguous).
    
    # Simulate price path for A
    A = B * np.exp(X)

    return A, B, X

class SystemParams:

    def __init__(self, s_0=100, rho_0=0.6, mu_i=0.0,
                 sigma_i=0.25, beta_i=-10.0, delta_i=1.0, a=0.0, b=0.0):

        self.s_0 = s_0
        self.rho_0 = rho_0
        self.mu_i = mu_i
        self.sigma_i = sigma_i
        self.beta_i = beta_i
        self.delta_i = delta_i
        self.a = a
        self.b = b

    def theta(self, mu_0, sigma_0):

        return -(self.b + mu_0 + self.beta_i*self.mu_i - 0.5*sigma_0**2
                 - 0.5*self.beta_i*self.sigma_i**2)/(self.beta_i*self.delta_i)

    @property
    def kappa(self):

        return -self.beta_i*self.delta_i


def simulate_benchmark_cointegrated_system(parameters, s0_0, mu_0, sigma_0, dt, corr_mat, n_step):

    if not isinstance(parameters, dict):
        raise ValueError('parameters have to be a dictionary.')

    n_assets = len(parameters)

    rho = np.zeros((corr_mat.shape[0]+1, corr_mat.shape[1]+1))
    rho[0, 0] = 1
    for i, (symbol, params) in enumerate(parameters.items()):
        rho[i+1, 0] = params.rho_0
        rho[0, i+1] = params.rho_0
    rho[1:, 1:] = corr_mat

    sigmas = np.zeros([n_assets + 1])
    sigmas[0] = sigma_0
    for i, (symbol, params) in enumerate(parameters.items()):
        sigmas[i+1] = params.sigma_i

    cov = np.matmul(np.matmul(np.diag(sigmas), rho), np.diag(sigmas))
    c = cholesky(cov)
    n = np.random.normal(0.0, 1.0, size=(n_step, n_assets+1))
    db = np.matmul(n, c)

    vol = np.std(db, axis=0)

    db_0 = db[:, 0]
    db_i = db[:, 1:]

    z = np.zeros((n_step, n_assets))
    ln_s_i = np.zeros((n_step, n_assets))
    ln_s_0 = np.zeros(n_step)

    ln_s_0[0] = np.log(s0_0)
    for j in range(1, n_step):
        d_ln_s_0 = (mu_0 - 0.5*sigma_0**2)*dt + np.sqrt(dt)*db_0[j]
        ln_s_0[j] = ln_s_0[j-1] + d_ln_s_0

    for i, (symbol, params) in enumerate(parameters.items()):

        z[0, i] = params.theta(mu_0, sigma_0)
        for j in range(1, n_step):
            dz = -params.beta_i*params.delta_i*(params.theta(mu_0, sigma_0) - z[j-1, i])*dt\
                 + np.sqrt(dt)*db_0[j] + params.beta_i*np.sqrt(dt)*db_i[j, i]
            z[j, i] = z[j-1, i] + dz

        ln_s_i[0, i] = (z[0, i] - params.a - params.b - ln_s_0[0])/params.beta_i
        for j in range(1, n_step):
            d_ln_s_i = (params.mu_i - 0.5*params.sigma_i**2 + params.delta_i*z[j-1, i])*dt \
                       + np.sqrt(dt)*db_i[j, i]
            ln_s_i[j, i] = ln_s_i[j-1, i] + d_ln_s_i

    return ln_s_0, ln_s_i, z