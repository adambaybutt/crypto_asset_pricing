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

# Function
# Purpose: perform lasso of Y on X
#
# Input args:
# -argument on whether to normalize the X's
# -lasso penalty
# -Y
# -X
# -look thru other arguments in Lasso function
#
# Output args:
# -beta hats
# 
# Steps inside function:
# -normalize X's if asked to do so
# -fit Lasso
# -Return the beta hats

'''
muhat = np.mean(X,axis = 0)
stdhat = np.std(X,axis = 0)
Xtilde = (X − muhat )/ stdhat
y = boston . target
lasso = Lasso ( alpha = 1.3)
lasso.fit(Xtilde ,y)
coef = lasso . coef_

sigma = np.std(y)
(n,p) = X.shape
Xscale = np.max(np.mean ((X ∗∗ 2), axis =0)) ∗∗ 0.5
c = 1.1; a = 0.05
lamb = 2 ∗ c ∗ sigma ∗ norm.ppf(1−a/(2 ∗ p)) ∗ Xscale /np.sqrt(n)
lamb1 = 2 ∗ c ∗ sigma ∗ norm.ppf(1−a/(2 ∗ p))/ np.sqrt(n)
print (lamb)
print (lamb1 )
'''


# Function 
# Purpose: perform OLS from scratch with closed form
#
# Input args:
# -Y
# -X
# -look thru other arguments in linear reg function to see if to specify anything
#
# Output args:
# -beta hats
# 
# Steps inside function:
# -fit
# -Return the beta hats

'''
n,p = df.shape
lnw = np.array (df["lnw"], ndmin = 2).T
female = np. array (df[" female "], ndmin = 2).T
lhs = np.array (df["lhs"], ndmin = 2).T
hsg = np.array (df["hsg"], ndmin = 2).T
sc = np.array (df["sc"], ndmin = 2).T
cg = np.array (df["cg"], ndmin = 2).T
cons = np.ones ([n ,1])
X = np. concatenate (( cons ,female ,lhs ,hsg ,sc ,cg), axis = 1)
Y = lnw
betahat = np. linalg .inv(X.T @ X) @ (X.T @ Y)
ehat = Y − X @ betahat
Sigmahat = (X ∗ ehat ).T @ (X ∗ ehat) / n
Qhat = np. linalg .inv(X.T @ X / n)
Vhat = Qhat @ Sigmahat @ Qhat
sdhat = np.sqrt(Vhat [1 ,1]) / np.sqrt(n)
cil = betahat [1] − 1.96 ∗ sdhat; cir = betahat [1] + 1.96 ∗ sdhat
'''

# Function
# Purpose: perform DL
#
# Input args:
# -set of controls of interest
# -Y
# -D
# -X
#
# Output args:
# -treatment parameter
# 
# Steps inside function:
# -lasso of Y on D and X to select elements
# -lasso of D on X to select elements
# -take the union of all three
# -OLS of Y on D plus that union
# -Return h t j which is parameter on D


# Function 
# Purpose: perform my estimation procedure
#
# Input args:
# -set of controls of interest
# -returns
# -G
# -Z
#
# Output args:
# -the vector of gmma beta 
#
# Steps inside function:
# -parallelize across j:
# --parallelize across each t, do DL for h t j
# --put all the h_t_j together for H_j
# --OLS of H on G to return each gamma beta j
# -put the gamma beta j together for single vector to return

# Function to run the simulation so call my estimation in parallel across all the simulations

# Main function to build the data and then call the function to run the simulation

# TODO:
# -Split out all my estimation code into a separate file (i.e. this file) and then just import those functions into a new file
# --read any formating on how to do this nicely