from __future__ import division
import numpy as np
from numpy.random import random
from numpy import newaxis as na
import scipy.stats as stats
import scipy.linalg

### Sampling functions

def sample_discrete(dist,size=[]):
    assert (dist >=0).all()
    cumvals = np.cumsum(dist)
    return np.sum(random(size)[...,na] * cumvals[-1] > cumvals, axis=-1)

def sample_niw(mu_0,lmbda_0,kappa_0,nu_0, chol=None):
    '''
    Returns a sample from the normal/inverse-wishart distribution, conjugate
    prior for (simultaneously) unknown mean and unknown covariance in a
    Gaussian likelihood model. Returns covariance.  '''
    # this is copied from www.mit.edu/~mattjj
    # reference: p. 87 in Gelman's Bayesian Data Analysis

    # first sample Sigma ~ IW(lmbda_0^-1,nu_0)
    # lmbda = np.linalg.inv(sample_wishart(np.linalg.inv(lmbda_0),nu_0))
    lmbda = sample_invwishart(lmbda_0,nu_0, chol) 
    # then sample mu | Lambda ~ N(mu_0, Lambda/kappa_0)
    mu = np.random.multivariate_normal(mu_0,lmbda / kappa_0)

    return mu, lmbda

def sample_invwishart(lmbda, dof, chol=None):
    # TODO make a version that returns the cholesky
    # TODO allow passing in chol/cholinv of matrix parameter lmbda
    n = lmbda.shape[0]
    if lmbda_chol is None:
        chol = np.linalg.cholesky(lmbda)

    if (dof <= 81+n) and (dof == np.round(dof)):
        x = np.random.randn(dof,n)
    else:
        x = np.diag(np.sqrt(stats.chi2.rvs(dof-(np.arange(n)))))
        x[np.triu_indices_from(x,1)] = np.random.randn(n*(n-1)/2)
    R = np.linalg.qr(x,'r')
    T = scipy.linalg.solve_triangular(R.T,chol.T).T
    return np.dot(T,T.T)

def sample_wishart(sigma, dof):
    '''
    Returns a sample from the Wishart distn, conjugate prior for precision matrices.
    '''

    n = sigma.shape[0]
    chol = np.linalg.cholesky(sigma)

    # use matlab's heuristic for choosing between the two different sampling schemes
    if (dof <= 81+n) and (dof == round(dof)):
        # direct
        X = np.dot(chol,np.random.normal(size=(n,dof)))
    else:
        A = np.diag(np.sqrt(np.random.chisquare(dof - np.arange(0,n),size=n)))
        A[np.tri(n,k=-1,dtype=bool)] = np.random.normal(size=(n*(n-1)/2.))
        X = np.dot(chol,A)

    return np.dot(X,X.T)
