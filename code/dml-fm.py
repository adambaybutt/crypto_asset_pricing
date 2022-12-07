import pandas as pd
import numpy as np

def DGP(S: int, N: int, T: int, 
        p: int, l: int, k: int, sparse: int, 
        rho: float, sigma_eps: float) -> tuple:
    ''' Generate random variables for simulation.

    Args: 
        S (int): number of simulations.
        N (int): number of observational units per time period.
        T (int): number of time periods.
        p (int): dimensionality of the covariates.
        l (int): dimensionality of observable factors.
        k (int): dimensionality of latent factors.
        sparse (int): sparsity index of factor loadings.
        rho (float): cross-sectional correlation of covariates.
        sigma_eps (float): standard deviation of idiosyncratic error.
    
    Returns: 
        (tuple):
            - R (np.ndarray): outcome/returns matrix of dimensions T*N, 1, S.
            - H (np.ndarray): true factor model of dimensions T, p, S.
            - G (np.ndarray): true observable factors of dimensions T, l, S.
            - F (np.ndarray): true latent factors of dimensions T, k, S.
            - e (np.ndarray): true idiosyncratic error of dimensions T*N, 1, S.
            - Gamma_beta (np.ndarray): true loading on observable factors of dim p by 1.
            - Gamma_delta (np.ndarray): true loading on latent factors of dim p by 1.
    '''
    # Initialize data objects 
    R = np.zeros((T*N, 1, S), dtype=float)
    Z = np.zeros((T*N, p, S), dtype=float)
    H = np.zeros((T, p, S), dtype=float)
    G = np.zeros((T, l, S), dtype=float)
    F = np.zeros((T, k, S), dtype=float)
    e = np.zeros((T*N, 1, S), dtype=float)

    # Initialize parameters
    Gamma_beta = np.zeros((p, 1))
    Gamma_delta = np.zeros((p, 1))
    for i in range(sparse):
        Gamma_beta[i, 0] = 1
        Gamma_delta[i, 0] = 1
    # TODO: FIX THE ABOVE TO MAKE THESE l and k matrices

    # Create Z covariance matrix
    Z_covar = np.zeros((p,p))
    for i in range(0,p):
        for j in range(0,p):
            Z_covar[i,j] = rho**(np.abs(i-j))

    # Generate data for each simulation
    for s in range(S):
        # Set seed for numpy and random packages for replicable data
        np.random.seed(s)

        # Form G
        phi_g = 0.5
        sigma_eps_g = 1
        g_eps       = np.random.normal(scale=sigma_eps_g, size=(T, l))
        G[0, :, s] = g_eps[0, :]
        for t in range(1, T):
            G[t, :, s] = phi_g*G[t-1, :, s] + g_eps[t, :]
        G[:, 0, s] = G[:, 0, s] - np.mean(G[:, 0, s])

        # Form F
        phi_f = 0.5
        sigma_eps_f = 1
        f_eps       = np.random.normal(scale=sigma_eps_f, size=(T,k))
        F[0, :, s] = f_eps[0,:]
        for t in range(1, T):
            F[t, :, s] = phi_f*F[t-1, :, s] + f_eps[t, :]
        F[:, :, s] = F[:, :, s] - np.mean(F[:, :, s], axis=0)

        # Form Z
        Z[:, :, s] = np.random.multivariate_normal(np.zeros(p), Z_covar, size=T*N)

        # Form idiosyncratic errors
        e[:,:,s] = np.random.multivariate_normal([0], [[sigma_eps**2]], size=T*N)

        # Form H
        # H[:,:,s] = G for each time for all l elements times matrix Gamma beta which is l times p + 
        #            F for each time for all the k terms times Gamma delta which is k times p 
        
        # Form outcome/returns
        # R[:,:,s] = Z times H where its each z_it of length p times ht of length p.
        
    return R, H, G, F, e, Gamma_beta, Gamma_delta

# Function to perform lasso; look how sklearn does it so i write it from scratch
# -go scope how Denis runs it in his class

# Function to perform OLS from scratch with closed form

# Function to run DL
# takes argument of set of controls of interest
# return the treatment parameter

# Function to run my procedure
# -call all the above functions
# return the vector of gmma beta j so parallize across the p runs of this within this function

# Function to run the simulation so call my estimation in parallel across all the simulations

# Main function to build the data and then call the function to run the simulation

# TODO:
# -Split out all my estimation code into a separate file (i.e. this file) and then just import those functions into a new file
# --read any formating on how to do this nicely