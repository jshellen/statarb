
import numpy as np

from .linear_algebra_tools import (
    givens_rotation
)


def kalman_filter_predict(X, P, A, Q, B, U):
    """
    Perform predict step
    """
    X = np.dot(A, X) + np.dot(B, U)
    P = np.dot(A, np.dot(P, A.T)) + Q
    return (X, P)


def kalman_filter_update(X, P, Y, H, R):
    """
    Perform update step
    """
    IM = np.dot(H, X)
    IS = R + np.dot(H, np.dot(P, H.T))
    # QR !
    Q, R = givens_rotation(IS)
    IS_inv = np.matmul(np.linalg.inv(R), Q.T)
    #IS_inv_svd = np.linalg.pinv(IS)
    K = np.dot(P, np.dot(H.T, IS_inv))
    X = X + np.dot(K, (Y - IM))
    P = P - np.dot(K, np.dot(IS, K.T))
    LH = gauss_pdf(Y, IM, IS)
    return (X, P, K, IM, IS, LH)


def gauss_pdf(X, M, S):


    if M.shape[1] == 1:

        DX = X - np.tile(M, X.shape[1])
        E = 0.5 * np.sum(DX * (np.dot(np.linalg.inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S))
        P = np.exp(-E)

    elif X.shape[1] == 1:

        DX = np.tile(X, M.shape()[1]) - M
        E = 0.5 * np.sum(DX * (np.dot(np.linalg.inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S))
        P = np.exp(-E)

    else:

        DX = X - M
        E = 0.5 * np.dot(DX.T, np.dot(np.inv(S), DX))
        E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(np.det(S))
        P = np.exp(-E)

    return (P[0], E[0])


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n_step = 500
    y_1 = np.cumprod(np.exp(0.05*np.sqrt(1.0/250)*np.random.normal(0, 1, n_step)))
    y_2 = 1.2*y_1 + 0.005*np.random.normal(0, 1, n_step)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(y_1)
    ax.plot(y_2)
    plt.show()

    # Initialization of state matrices
    X = np.array([[1.0, 1.0]]).reshape(-1, 1)
    delta = 1e-5
    P = delta / (1 - delta) * np.eye(2)
    A = np.array([[1, 0],
                  [0, 1]])
    Q = np.zeros(X.shape)
    B = np.zeros(X.shape).reshape(1, -1)
    U = np.zeros((X.shape[0], 1))

    # Measurement matrices

    R = 0.000000000001

    X_t = np.zeros((n_step, 2))

    for i in np.arange(0, n_step):
        Y = np.array([[y_2[i]]])
        H = np.array([y_1[i], 1]).reshape(1, -1)
        (X, P) = kalman_filter_predict(X, P, A, Q, B, U)
        (X, P, K, IM, IS, LH) = kalman_filter_update(X, P, Y, H, R)
        X_t[i, :] = X.flatten()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(X_t)
    plt.show()
