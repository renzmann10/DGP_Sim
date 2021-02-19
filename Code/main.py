'''
Data Generating Process Simulator

Author: Roman Enzmann ((c))
'''

# Import libraries
# Standard
import os
import numpy as np

# Random number generators
import random
from random import randrange


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
#                                           DEFINITIONS 
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------


def dgp_generator(classification, class_prob, num_nonlinear, p, highest_pd, seed = 0, constant = True, n = 2000,\
                    num_irrelevant = None, num_correlated = None, noise_lvl=1.0,\
                    noise_dist = 'gauss_standard'):
    '''
    Draws a random sample tuple, by default, from a Gauss normal distribution with mu = 0 and std = std(y)
    EXAMPLE: X_testingdgp, y_testingdgp = dgp_generator(classification = True, class_prob = [0.3,0.7],
                                            num_nonlinear = 4, p = 10, highest_pd = 5)

    Inputs:
    --------------
    classification : boolean
        if True --> y is binary
        o/w     --> y is a float
    class_prob : 2d array
        defines the true population probability of the classes
    constant : boolean (DEFAULT = True)
        True
    p : int
        number of features
    n : int (DEFAULT = 2000)
        sample size
    num_nonlinear : int
        number of non-linear features
    num_irrelevant, OPTIONAL : int (DEFAULT = None)
        number of irrelevant features
    num_correlated, OPTIONAL: int (DEFAULT = None)
        number of correlated features
    highest_pd : int
        highest polynomial degree
    noise_dist : str (DEFAULT = 'std_norm')
        PDF of DGP, scaled by standard deviation of response
            available strings:  i) 'gauss_standard' : standard distribution
                                ii) 'gauss_fat'
                                iii) 'gamma' : left shift of mean
    seed : int (DEFAULT = 0)
        random seed

    Outputs:
    --------------

    length-n synthetic data set
    
    '''
    np.random.seed(seed)
    if num_nonlinear == 0:
        highest_pd = 1
    if constant == True:
        constant = 1
    else:
        constant = 0
    if num_irrelevant is not None:
        num_info = p-num_irrelevant
    else:
        num_info = p
    # Initialise
    y = np.zeros(n)
    X = np.zeros(n*p).reshape(n,p)
    weights = np.random.uniform(-2, 2, num_info)
    powers = [1 for item in range(num_info)]
    # Features
    if num_correlated == None:
        for j in range(X.shape[1]):
            mu = np.random.randint(-5,5,1)
            std = np.random.randint(0,5,1)
            X[:, j] = np.random.normal(mu, std, n)
    else:
        non_corr = num_info-num_correlated
        mu = np.array([int(np.random.randint(-5,5,1)) for item in range(num_correlated)])
        cov_matrix = np.random.uniform(-5, 5, (num_correlated*num_correlated)).\
        reshape(num_correlated, num_correlated)
        for q in range(num_correlated):
            X[:, q] = np.random.multivariate_normal(mu, cov_matrix, n)
        for j in range(non_corr):
            mu = np.random.randint(-5,5,1)
            std = np.random.randint(0,5,1)
            X[:, j] = np.random.normal(mu, std, n)
    # Weights
    if num_nonlinear > 0:
        powers_val = np.random.randint(1, (highest_pd+1), num_nonlinear)
        indic = np.random.randint(1, (num_info), num_nonlinear)
        for q in indic:
            indic = np.random.randint(1,num_nonlinear,1)
            powers[q] = powers_val[indic]
        for j in indic:
            X[:,j] = X[:,j]**powers[j]
    # before noise
    X_t = np.transpose(X)
    y = constant + np.dot(weights,X_t)

    # Noise
    if noise_dist == 'gauss_standard':
        pdf = 'normal'
        mu = 0
        sigma = np.std(y,ddof=1)
        eps = getattr(np.random,pdf)(mu,sigma,n)
    elif noise_dist == 'gauss_fat':
        pdf = 'normal'
        mu = 0
        sigma = np.std(y,ddof=1)*5 # for fat tails
        eps = getattr(np.random,pdf)(mu,sigma,n)
    elif noise_dist == 'gamma':
        pdf = 'gamma'
        shape = 1
        scale = np.std(y,ddof=1)
        eps = getattr(np.random,pdf)(shape,scale,n)
    else:
        ValueError('not properly specified')
        print('value not properly specified')
    # After noise
    y_obs = y+eps

    if  classification == True:
        min_prob = min(class_prob)
        q_yclassprob = np.quantile(y_obs,min_prob)
        for i in range(len(y_obs)):
            if abs(y_obs[i]) >= q_yclassprob:
                y_obs[i] = 1
            else:
                y_obs[i] = 0
    else:
        y_obs = y_obs
    
    return X, y_obs;
