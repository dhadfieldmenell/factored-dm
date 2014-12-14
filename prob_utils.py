from __future__ import division

import numpy as np
import scipy.stats as ss
from numpy.linalg import cholesky as chol

import copy, sys, h5py
import IPython as ipy

from stats_util import sample_niw

eps = 1e-4

################################################################################
## Probability Utils
################################################################################

class Dist(object):
    
    def sample(self, N):
        return self._sample(N)

    def quantile(self, v, exact=False, N=200):
        """
        @returns: q 
        a value for the mean such that x is the proportion of
        distributions with mu < q
        """
        if exact:
            return bisect1d(v, 0, 1, self.cdf)
        means = np.array(sorted([d.mean() for d in self.sample(N)]))        
        min_q = int(np.floor(v * N))
        try:
            return (means[min_q] + means[min_q + 1]) / 2
        except IndexError:
            return means[min_q]

class Prior(Dist):

    def update(self, o):
        raise NotImplemented

class Beta(Prior):
    
    def __init__(self, (alpha, beta)):
        super(Beta, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def _sample(self, N):
        thetas = np.random.beta(self.alpha, self.beta, N)
        return [Bernoulli(t) for i, t in enumerate(thetas)]

    def update(self, o):
        return Beta((self.alpha + o, self.beta + 1-o))

    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    def cdf(self, theta):
        return ss.beta.cdf(theta, self.alpha, self.beta)

    def __hash__(self):
        return hash((self.alpha, self.beta))

class Dirichlet(Prior):
    
    def __init__(self, theta):
        super(Dirichlet, self).__init__()
        self.theta = theta

    def _sample(self, N):
        return np.random.dirichlet(self.theta, N)

    def mean(self):
        return self.theta / np.sum(self.theta)

    def __hash__(self):
        tmp = self.theta.flags.writeable
        self.theta.flags.writeable=False
        h = hash(self.theta.data)
        self.theta.flags.writeable=tmp
        return h

class Multinomial(Dist):
    
    def __init__(self, pi):
        super(Multinomial, self).__init__()
        self.pi = pi

    def _sample(self, N):
        return np.random.multinomial(1, self.w, size=N)

    def mean(self):
        return self.pi

class Bernoulli(Dist):
    
    def __init__(self, theta):
        super(Bernoulli, self).__init__()
        self.theta = theta
        
    def mean(self):
        return self.theta

    def _sample(self, N):
        return np.random.binomial(1, self.theta, N)

class NormalInverseGamma(Prior):
    def __init__(self, mu, nu, alpha, beta):
        super(NormalInverseGamma, self).__init__()
        self.mu = mu
        self.nu = nu
        self.alpha = alpha
        self.beta = beta

    def mean(self):
        return self.mu
    
    def _sample(self, N):
        result = []
        for _ in range(N):
            tau = np.random.gamma(self.alpha, self.beta)
            sigma_sq = 1/tau
            result.append(Gaussian(np.random.normal(self.mu, sigma_sq / self.nu), np.sqrt(sigma_sq)))
        return np.asarray(result)

    def update(self, o):
        nu_next = self.nu+1
        mu_next = (self.nu * self.mu + o) / (nu_next)
        alpha_next = self.alpha + 1/2
        beta_next = self.beta + (self.nu  * (o - self.mu)**2)/ (2*nu_next)
        return NormalInverseGamma(mu_next, nu_next, alpha_next, beta_next)

class Gaussian(Dist):
    
    def __init__(self, mu, sigma):
        super(Gaussian, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def mean(self):
        return self.mu

    def _sample(self, N):
        return np.random.normal(self.mu, self.sigma, N)

class NormalInverseWishart(Prior):
    def __init__(self, mu, lmbda, kappa, nu):
        super(NormalInverseWishart, self).__init__()

        self.mu = mu
        self.lmbda = lmbda
        self.nu = nu
        self.kappa = kappa
        self.lmbda_chol = chol(lmbda)
        self.d = mu.shape[0]

    def mean(self):
        return self.mu, self.lmbda 
        
    def _sample(self, N):
        return sample_niw(
            self.mu, self.lmbda, self.kappa, self.nu, self.lmbda_chol)

    def update(self, o):
        kappa_next = self.kappa + 1
        nu_next = self.nu + 1
        mu_next = (self.kappa * self.mu + o) / (kappa_next)
        diff = o - self.mu
        lmbda_next = self.lmbda + (self.kappa / (kappa_next)) * diff.dot(diff.T)
        return  NormalInverseWishart(mu_next, lmbda_next, kappa_next, nu_next)

class MultivariateGaussian(Dist):
    
    def __init__(self, mu, sigma):
        super(MultivariateGaussian, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.d = self.mu.shape[0]

    def mean(self):
        return self.mu

    def _sample(self, N):
        return np.random.multivariate_normal(self.mu, self.sigma, N)

class ParameterFilter(Prior):
    ## TODO
    pass

    
def bisect1d(targ, xmin, xmax, f, tol=eps):
    xmid = (xmax - xmin) / 2
    fmid = f(xmid)
    while np.abs(fmid - targ) > tol:
        if fmid < targ:
            xmin = xmid
        else:
            xmax = xmid
        xmid = (xmax + xmin) / 2
        fmid = f(xmid)
    return xmid

