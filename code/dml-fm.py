import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from scipy.stats import norm
from joblib import Parallel, delayed

def DGP(S: int, N: int, T: int, 
        p: int, l: int, k: int, sparse: int, 
        rho: float, sigma_eps: float) -> tuple:
    ''' Generate random variables and parameters for simulation.

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
            - R (np.ndarray): outcome/returns ndarray of dimensions T*N, 1, S.
            - Z (np.ndarray): covariates of dimensions T*N, p+1, S.
            - H (np.ndarray): true factor model of dimensions T, p+1, S.
            - G (np.ndarray): true observable factors of dimensions T, l, S.
            - F (np.ndarray): true latent factors of dimensions T, k, S.
            - e (np.ndarray): true idiosyncratic error of dimensions T*N, 1, S.
            - Gamma_beta (np.ndarray): true loading on observable factors of dim (p+1) by 1.
            - Gamma_delta (np.ndarray): true loading on latent factors of dim (p+1) by 1.
    '''
    # Initialize data objects 
    R = np.zeros((T*N, 1, S), dtype=float)
    Z = np.zeros((T*N, p+1, S), dtype=float)
    H = np.zeros((T, p+1, S), dtype=float)
    G = np.zeros((T, l, S), dtype=float)
    F = np.zeros((T, k, S), dtype=float)
    e = np.zeros((T*N, 1, S), dtype=float)

    # Initialize parameters
    Gamma_beta = np.zeros((p+1, l))
    Gamma_delta = np.zeros((p+1, k))
    for i in range(sparse):
        for j in range(l):
            Gamma_beta[i, j] = 1
        for j in range(k):
            Gamma_delta[i, j] = 1

    # Create Z covariance ndarray
    Z_covar = np.zeros((p+1,p+1))
    for i in range(0,p+1):
        for j in range(0,p+1):
            Z_covar[i,j] = rho**(np.abs(i-j))

    # Generate data for each simulation
    for s in range(S):
        # Set seed for numpy and random packages for replicable data
        np.random.seed(s)

        # Form G
        phi_g = 0.8
        sigma_eps_g = 0.1
        g_eps       = np.random.normal(scale=sigma_eps_g, size=(T, l))
        G[0, :, s] = g_eps[0, :]
        for t in range(1, T):
            G[t, :, s] = phi_g*G[t-1, :, s] + g_eps[t, :]
        G[:, 0, s] = G[:, 0, s] - np.mean(G[:, 0, s])

        # Form F
        phi_f = 0.7
        sigma_eps_f = 0.1
        f_eps       = np.random.normal(scale=sigma_eps_f, size=(T,k))
        F[0, :, s] = f_eps[0,:]
        for t in range(1, T):
            F[t, :, s] = phi_f*F[t-1, :, s] + f_eps[t, :]
        F[:, :, s] = F[:, :, s] - np.mean(F[:, :, s], axis=0)

        # Form Z
        Z[:, :, s] = np.random.multivariate_normal(np.zeros(p+1), Z_covar, size=T*N)

        # Form idiosyncratic errors
        e[:,:,s] = np.random.multivariate_normal([0], [[sigma_eps**2]], size=T*N)

        # Form H
        H[:,:,s] = np.matmul(G[:,:,s], np.transpose(Gamma_beta)) + \
                    np.matmul(F[:,:,s], np.transpose(Gamma_delta))

        # Form outcome/returns
        R[:,:,s] = np.sum(Z[:,:,s]*np.repeat(H[:,:,s], N, axis=0),axis=1).reshape(-1,1) + e[:,:,s]

    return R, Z, H, G, F, e, Gamma_beta, Gamma_delta

def runLasso(Y: np.ndarray, X: np.ndarray, penalty: float) -> np.ndarray:
    ''' Runs lasso of Y on X with given penalty param to return fitted coefs.

    Args: 
        X (np.ndarray): RHS variables with rows of obs and cols of covars.
                       These data include a constant but have yet to be
                       normalized for lasso.
        Y (np.ndarray): LHS variable with rows of obs and single column.
        penalty (float): real-valued scalar on L1 penalty in Lasso.

    Returns:
        beta_hat (np.ndarray): vector of fitted coefficients; note: these
                             must be used on normalized RHS variables.
    '''
    # normalize RHS
    muhat  = np.mean(X, axis = 0)
    stdhat = np.std(X, axis = 0)
    Xtilde = np.divide(np.subtract(X, muhat), stdhat)

    # perform lasso
    lasso = Lasso(alpha = penalty)
    lasso.fit(Xtilde, Y)

    # return fitted coefficients
    return lasso.coef_

def calcPenaltyBCCH(Y: np.ndarray, X: np.ndarray, c: float) -> float:
    ''' This function applies Belloni, Chen, Chernozhukov, Hansen 2012 ECMA
        closed-form solution for selecting Lasso penalty parmaeter.

    Args: 
        X (np.ndarray): RHS variables with rows of obs and cols of covars.
                       These data include a constant but have yet to be
                       normalized for lasso.
        Y (np.ndarray): LHS variable with rows of obs and single column.
        c (float):    scalar constant from theory; usually ~1.

    Returns:
        penalty (float): BCCH penalty parameter.
    '''
    # Bickel Ritov Tsybakov constant parameter selection
    a = 0.1

    # calc pilot penalty parameter
    N = X.shape[0]
    p = X.shape[1]
    max_moment_xy = np.max(np.mean((X**2)*(Y**2), axis =0)**0.5) 
    penalty_pilot = 2*c*norm.ppf(1-a/(2*p))*max_moment_xy/np.sqrt(N)

    # run lasso with pilot penalty parameter
    beta_hat = runLasso(Y, X, penalty_pilot)
    assert(~np.isclose(0, np.sum(np.abs(beta_hat)), rtol=1e-8, atol=1e-8)),('Pilot penalty kills all coefs. Scale down c?')

    # set BCCH penalty parameter
    residuals = Y - np.matmul(X, beta_hat).reshape(-1,1)
    max_moment_xepi = np.max(np.mean((X**2)*(residuals**2), axis =0)**0.5) 
    penalty = 2*c*norm.ppf(1-a/(2*p))*max_moment_xepi/np.sqrt(N)

    return penalty

def runOLS(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    ''' Runs OLS of Y on X to return fitted coefficients.

    Args: 
        X (np.ndarray): RHS--assumes contains constant--with rows of obs and cols of covars.
        Y (np.ndarray): LHS variable with rows of obs and single column.

    Returns:
        beta_hat (np.ndarray): vector of fitted coefficients.
    '''
    return np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)),
                     np.matmul(np.transpose(X), Y))

def runDoubleLasso(Y: np.ndarray, D: np.ndarray, X: np.ndarray,
                   amel_set: list, c: float) -> float:
    ''' Runs Double Selection Lasso from Belloni et al (2014).

    Args: 
        Y (np.ndarray): LHS variable with rows of obs and single column.
        D (np.ndarray): RHS target variable with rows of obs and single column.
        X (np.ndarray): RHS controls with rows of obs and cols of covars.
        amel_set (list): column indices of X to manually include in final OLS reg,
                         termed the amelioration set.
        c (float):    scalar constant from theory; usually ~1.

    Returns:
        alpha_hat (float): estimated target coefficient.
    '''
    # lasso of Y on D and X to select elements of X, I_1_hat
    X_all = np.hstack((D,X))
    beta_hat_1 = runLasso(Y, X_all, penalty=calcPenaltyBCCH(Y, X_all, c=c))

    # lasso of D on X to select elements of X, I_2_hat
    beta_hat_2 = runLasso(D, X, penalty=calcPenaltyBCCH(D, X, c=c))

    # form union of I_1_hat, I_2_hat, and amel_set
    i_1_hat = list(np.nonzero(beta_hat_1)[0]-1)
    if -1 in i_1_hat: i_1_hat.remove(-1) # remove treatment variable if it was included
    i_2_hat = list(np.nonzero(beta_hat_2)[0])
    i_3_hat = amel_set
    i_hat   = list(set(i_1_hat).union(set(i_2_hat), set(i_3_hat)))
    print('Double Selection Lasso is using '
          +str(int(len(i_hat)/X.shape[1]*100))
          +'% of the columns of the controls.')

    # OLS of Y on D plus included Xs
    X_sel    = X[:,i_hat]
    X_all    = np.hstack((D,X_sel))
    beta_hat = runOLS(Y,X_all)
    alpha_hat = beta_hat[0,0]

    # return target parameter on D
    return alpha_hat

def runEstimation(R: np.ndarray, Z: np.ndarray, G: np.ndarray, 
                  amel_set: list, c: float) -> np.ndarray:
    ''' This function performs the estimation procedure from Baybutt (2022).

    Args: 
        R (np.ndarray):  outcome/returns ndarray of dimensions T*N by 1.
        Z (np.ndarray):  covariates of dimensions T*N by (p+1).
        G (np.ndarray):  true observable factors of dimensions T by l.
        amel_set (list): column indices of X to manually include in final OLS reg,
                         termed the amelioration set.
        c (float):       scalar constant from theory; usually ~1.

    Returns:
        Gamma_beta_hat (np.ndarray): estimated loading on observable factors of dim (p+1) by 1.
    '''
    # initialize objects
    T = G.shape[0]
    N = int(R.shape[0]/T)
    p = int(Z.shape[1]-1)
    l = G.shape[1]
    H_hat = np.zeros((T, p+1), dtype=float)
    
    def runForEachCovariate(j):
        # form indices
        minus_j = list(range(p+1))
        minus_j.remove(j)
  
        def runForEachTimePeriod(t):
            # form indices
            start_ob = int(t*N)
            last_ob  = int((t+1)*N)

            # form this time periods LHS, target, and controls
            Y = R[start_ob:last_ob,:]
            D = Z[start_ob:last_ob,j].reshape(-1,1)
            X = Z[start_ob:last_ob,minus_j]

            # estimate h_{t,j}, i.e. target coef
            h_t_j = runDoubleLasso(Y, D, X, amel_set, c)

            # save h_{t,j}
            return h_t_j

        # estimate h_{t,j} for all time periods and save
        H_t_hat    = Parallel(n_jobs=2)(delayed(runForEachTimePeriod)(t) for t in range(T))
        H_hat[:,j] = np.array(H_t_hat)

        # estimate 
        return runOLS(H_hat[:,j], G)

    # estimate \Gamma_{\beta,j} using all time periods for all j
    Gamma_beta_hat = Parallel(n_jobs=2)(delayed(runForEachCovariate)(j) for j in range(p+1))

    return np.array(Gamma_beta_hat)

def runSimulation(R: np.ndarray, Z: np.ndarray, G: np.ndarray,
                  amel_set: list, c: float, num_cpus: int) -> tuple:
    ''' Runs the simulation to estimate target parameters and return estimates across various
        procedures for all monte carlo repetitions.

    Args: 
        R (np.ndarray):  outcome/returns ndarray of dimensions T*N by 1.
        Z (np.ndarray):  covariates of dimensions T*N by (p+1).
        G (np.ndarray):  true observable factors of dimensions T by l.
        amel_set (list): column indices of X to manually include in final OLS reg,
                         termed the amelioration set.
        c (float):       scalar constant from theory; usually ~1.
        num_cpus (int):  number of cpus on user machine.
    
    Returns: 
        (tuple):
            - Gamma_beta_hat_b (np.ndarray): estimated gamma beta for my proc of dim (p+1) by S.
            - Gamma_beta_hat_ipca (np.ndarray): estimated gamma beta for ipca of dim (p+1) by S.
    '''
    def runForEachMCrep(s):
        return runEstimation(R[:,:,s], Z[:,:,s], G[:,:,s], amel_set, c)
    
    # estimate \Gamma_\beta for each Monte Carlo repetition and clean up shape
    # note: runEstimation will split each call across four cpus hince divide num_cpus by 4
    Gamma_beta_hat_b = Parallel(n_jobs=int(num_cpus/4))(delayed(runForEachMCrep)(s) 
                                                                for s in range(S))
    Gamma_beta_hat_b = np.transpose(np.array(Gamma_beta_hat_b)[:,:,0])

    return (Gamma_beta_hat_b, )

def calcMSE(est: np.ndarray, param: np.ndarray) -> float:
    ''' Calculate the mse between an estimator and parameter.
    
    Args:
        est (np.ndarray): estimator of length of parameter by S.
        param (np.ndarray): vector of parameter with single column.

    Returns: 
        mse (float): mean squared estimation error.
    '''
    return np.mean(np.square(est - param), axis=1)

def calcBias2(est: np.ndarray, param: np.ndarray) -> float:
    ''' Calculate the squared bias of estimation error between an estimator and parameter.
    
    Args:
        est (np.ndarray): estimator of length of parameter by S.
        param (np.ndarray): vector of parameter with single column.

    Returns: 
        bias2 (float): squared bias of estimation error.
    '''
    return np.square(np.mean(est, axis=1).reshape(-1,1) - param)

def calcVar(est: np.ndarray, param: np.ndarray) -> float:
    ''' Calculate the variation of estimation error between an estimator and parameter.
    
    Args:
        est (np.ndarray): estimator of length of parameter by S.
        param (np.ndarray): vector of parameter with single column.

    Returns: 
        bias2 (float): variance of estimation error.
    '''
    return np.square(est - np.mean(est, axis=1).reshape(-1,1))

def calcEstimationErrors(Gamma_beta: np.ndarray, 
                         Gamma_beta_hat_b: np.ndarray) -> tuple:
    ''' Calculate estimation errors for procedures across simulations.
    
    Args:
        Gamma_beta (np.ndarray): true loading on observable factors of dim (p+1) by 1.
        Gamma_beta_hat_b (np.ndarray): estimated gamma beta for my proc of dim (p+1) by S.
        Gamma_beta_hat_ipca (np.ndarray): estimated gamma beta for ipca of dim (p+1) by S.

    Returns:
        (tuple):
            - mse_b (float): estimated mse of my proc across simulations.
            - bias2_b (float): estimated bias^2 of my proc across simulations.
            - var_b (float):  estimated variance of my proc across simulations.
            - mse_ipca (float): estimated mse of ipca across simulations.
            - bias2_ipca (float): estimated bias^2 of ipca across simulations.
            - var_ipca (float):  estimated variance of ipca across simulations.
    '''
    # calculate estimation errors of my procedure
    mse_b   = np.mean(calcMSE(Gamma_beta_hat_b, Gamma_beta))
    bias2_b = np.mean(calcBias2(Gamma_beta_hat_b, Gamma_beta))
    var_b   = np.mean(calcVar(Gamma_beta_hat_b, Gamma_beta))

    return (mse_b, bias2_b, var_b, )

if __name__ == "__main__":
    # set simulation parameters
    num_cpus  = 20
    amel_set  = [1]
    c         = 0.1
    rho       = 0.1
    sigma_eps = 1

    S         = 10 # TODO CHANGE BACK TO 200
    N         = 200
    T         = 200
    p         = 20
    l         = 1 
    k         = 2
    sparse    = 5

    # build DGP
    R, Z, H, G, F, e, Gamma_beta, Gamma_delta =  DGP(S, N, T, p, l, k, sparse, rho, sigma_eps)

    # run simulation
    Gamma_beta_hat_b, = runSimulation(R, Z, G, amel_set, c, num_cpus)

    # calc estimation errors
    mse_b, bias2_b, var_b, = calcEstimationErrors(Gamma_beta, Gamma_beta_hat_b)

    # report
    print('estimation errors for my procedure:')
    print(f'bias2: {bias2_b}')
    print(f'var: {var_b}')
    print(f'mse: {mse_b}')
    print('estimation errors for ipca:')

# TODO:
# -Split out all my estimation code into a separate file (i.e. this file) and then just import those functions into a new file
# --read any formating on how to do this nicely