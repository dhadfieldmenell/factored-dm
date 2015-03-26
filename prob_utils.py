from __future__ import division

import numpy as np
import scipy.stats as ss
from numpy.linalg import cholesky as chol

import copy, sys, h5py
import IPython as ipy

from stats_util import sample_niw
import warnings

eps = 1e-4

################################################################################
## Probability Utils
################################################################################

class Dist(object):

    exact_quantile = False
    
    def sample(self, N=1):
        return self._sample(N)

    def quantile(self, v, N=200):
        """
        @returns: q 
        a value for the mean such that x is the proportion of
        distributions with mu < q
        """
        if self.exact_quantile:
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

    exact_quantile = True
    
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
    def __init__(self, mu, nu, alpha, beta, q=0):
        super(NormalInverseGamma, self).__init__()
        self.mu = mu
        self.nu = nu
        self.alpha = alpha
        self.beta = beta
        self.q = q

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
        q_next = self.q + np.power(o, 2)
        return NormalInverseGamma(mu_next, nu_next, alpha_next, beta_next, q_next)

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

class SumDist(Dist):
    
    def __init__(self, *dists):
        self.dists = dists

    def _sample(self, N):
        return np.sum(d.sample(N) for d in self.dists)

    def mean(self):
        return np.sum(d.mean() for d in self.dists)

class DiracDelta(Dist):
    
    def __init__(self, v):
        self.v = v

    def _sample(self, N):
        return np.ones(N) * self.v

    def mean(self):
        return self.v

class LinNormal(Prior):
    
    def __init__(self, kappa_sq, sigma_sq, d):
        self.d = d
        self.kappa_sq = kappa_sq
        self.sigma_sq = sigma_sq
        self.ratio = self.sigma_sq / self.kappa_sq
        self.cov  = np.eye(d)*self.kappa_sq
        self.mu = np.zeros(d)
        # design matrix
        self.X_td = None
        self.Y_t = None

    def update(self, (y, U)):
        if self.X_td is None:
            self.X_td = U[None, :]
            self.Y_t  = np.array([y])
        else:
            self.X_td = np.r_[self.X_td, U[None, :]]
            self.Y_t  = np.r_[self.Y_t, [y]]
        
        tmp = np.linalg.inv(self.X_td.T.dot(self.X_td) + 
                            self.ratio * np.eye(self.d))
        self.mu = tmp.dot(self.X_td.T.dot(self.Y_t))
        self.cov  = tmp * self.sigma_sq
        ## return self rather than a copy
        ## so that we use the same object for
        ## all arms
        return self

    def mean(self):
        return self.mu
    
    # @profile
    def sample(self, N=1, U=None):
        thetas = np.random.multivariate_normal(self.mu, self.cov, N)
        if U is None:
            return thetas
        return [Gaussian(t.dot(U), np.sqrt(self.sigma_sq)) 
                for t in thetas]

    def sample_init(self):
        return self.sample()

class SpikeSlab(Prior):

    C_N = 100
    burn_in = 50
    gibbs_rate = 20
    _cache_size = 300
    
    MAX_EXACT_D = 10

    def __init__(self, eps, kappa_sq, sigma_sq, d, 
                 sparsity_pattern=None):
        super(SpikeSlab, self).__init__()
        self.eps = eps
        self.kappa_sq = kappa_sq
        self.sigma_sq = sigma_sq
        self.d = int(d)
        self.use_gibbs = d > self.MAX_EXACT_D        

        if self.use_gibbs:
            self.gibbs_rate = d*2
            self.burn_in    = d*5

        self.ratio = self.sigma_sq / self.kappa_sq
        # design matrix
        self.X_td = None
        self.Y_t = None
        self.C_hat = None
        self._cache = {}

        self.likelihoods = []
        self._cache_hits = 0

        if not self.use_gibbs:
            self._C_pdf = np.zeros(np.power(2, self.d))
            self._C_vals = get_binary_vals(self.d)

        if sparsity_pattern is None:
            self.sparsity_pattern = np.zeros(self.d)
            for i in range(np.round(eps*self.d)):
                self.sparsity_pattern[i] = 1
        else:
            self.sparsity_pattern = sparsity_pattern

    def sample_init(self):
        theta = np.random.multivariate_normal(
            np.zeros(self.d), self.kappa_sq * np.eye(self.d))
        theta[self.sparsity_pattern < 0.5] = 0
        return theta

    def update(self, (y, U), resample=True):
        if self.X_td is None:
            self.X_td = U[None, :]
            self.Y_t  = np.array([y])
        else:
            self.X_td = np.r_[self.X_td, U[None, :]]
            self.Y_t  = np.r_[self.Y_t, [y]]
        if resample:
            self.resample_C()        
        return self

    # @profile
    def resample_C(self):
        if self.use_gibbs:
            self.gibbs_sample_C()
        else:
            self._cache = {}
            C = np.zeros(self.d)
            for i in xrange(np.power(2, self.d)):
                C = self._C_vals[i, :]
                Gamma, Sigma_det = self.get_mvn_params(C)
                self._C_pdf[i] = self.log_p(C)
            log_Z = logsumexp(*self._C_pdf)
            self._C_pdf = np.exp(self._C_pdf - log_Z)
            
            

    #@profile
    def sample(self, N=1, U=None, thetas=None):
        if thetas is None:
            thetas = np.zeros((N, self.d))
            ones = self.sample_ones(N)
            with warnings.catch_warnings():
                # b/c we get warnings from np.random.multivariate_normal
                # for some weird reason
                warnings.simplefilter('ignore')
                for i in range(N):
                    thetas[i, ones[i]] = self.sample_theta(ones[i])
        if U is None:
            return thetas
        return [Gaussian(t.dot(U), np.sqrt(self.sigma_sq)) 
                for t in thetas]

    def sample_theta(self, C):
        nnz = int(np.sum(C))
        if nnz == 0:
            return None
        if self.X_td is None:
            mu = np.zeros(nnz)
            cov = self.kappa_sq * np.eye(nnz)
        else:
            data_mat = self.X_td[:, C]
            tmp = np.linalg.inv(data_mat.T.dot(data_mat) + 
                                (self.sigma_sq/self.kappa_sq) * np.eye(nnz))
            mu = tmp.dot(data_mat.T.dot(self.Y_t))
            cov = self.sigma_sq * tmp
        return np.random.multivariate_normal(mu, cov)

    # @profile
    def sample_ones(self, N=1):
        if self.X_td is None:
            ones = np.random.rand(N, self.d) > self.eps

        elif self.use_gibbs:
            ridx = np.random.randint(len(self.C_hat), size=N)
            ones = np.zeros((N, self.d), dtype=np.bool)
            for i in xrange(N):
                # to convert to a boolean
                ones[i, :] = self.C_hat[ridx[i]] > 0.5
        else:
            options = range(self._C_pdf.shape[0])
            inds = np.random.choice(options, p=self._C_pdf, size=N)
            ones = np.zeros((N, self.d), dtype=np.bool)
            for i, idx in enumerate(inds):
                ones[i, :] = self._C_vals[idx, :] > 0.5
        return ones

    def mean_C(self):
        if self.use_gibbs:
            return np.sum(self.C_hat, axis = 0) / len(self.C_hat)
        else:
            mu = np.zeros(self.d)
            for i in xrange(np.power(2, self.d)):
                mu += self._C_pdf[i] * self._C_vals[i]
            return mu            

    def gibbs_sample_C(self):
        self._cache={}
        self._cache_hits = 0
        self.likelihoods = []
        if self.C_hat is not None:
            # initialize with a sample from previous iteration
            ridx = np.random.randint(len(self.C_hat))
            C = np.copy(self.C_hat[ridx]).astype(np.int)
        else:
            # sample from prior
            C = (np.random.rand(self.d) > self.eps).astype(np.int)
        # C = self.C_init
        Gamma, Sigma_det = self.get_mvn_params(C)
        log_pc = self.log_p(C, Gamma, Sigma_det)

        self.C_hat = []
        for _ in range(self.burn_in):
            C, Gamma, Sigma_det, log_pc = self.gibbs_step(C, Gamma, Sigma_det, log_pc)            
        
        idx = 0
        for _ in range(self.C_N):            
            for _ in range(self.gibbs_rate):
                C, Gamma, Sigma_det, log_pc = self.gibbs_step(C, Gamma, Sigma_det, log_pc, idx = idx)
                idx = (idx + 1) % self.d

            self.C_hat.append(np.copy(C))

    def _cache_insert(self, key, data):
        if len(self._cache) >= self._cache_size:
            to_del = self._cache.keys()[np.random.randint(self._cache_size)]
            del self._cache[to_del]
        self._cache[key] = data

    # @profile
    def gibbs_step(self, C, Gamma, Sigma_det, log_pc, idx = None):
        """
        Does a gibbs step on the vector of non-zero
        indices for a spike-slab posterior, as described
        by Kaufmann et al.
        """
        if idx is None:
            idx = np.random.randint(self.d)        
        u = self.X_td[:, idx][:, None]
        idx_val = C[idx]
        if idx_val:
            Gamma_1 = Gamma
            Sigma_det_1 = Sigma_det
            log_p1 = log_pc
            C[idx] = 0
            key = tuple(C.tolist())
            if key in self._cache:
                self._cache_hits += 1
                Gamma_0, Sigma_det_0, log_p0 = self._cache[key]
            else:
                Gamma_0, Sigma_det_0 = sherman_morrison_update(
                    Gamma, Sigma_det, -u, u)
                log_p0 = self.log_p(C, Gamma_0, Sigma_det_0)
                self._cache_insert(key, (Gamma_0, Sigma_det_0, log_p0))
        else:
            Gamma_0 = Gamma
            Sigma_det_0 = Sigma_det
            log_p0 = log_pc
            C[idx] = 1
            key = tuple(C.tolist())
            if key in self._cache:
                self._cache_hits += 1
                Gamma_1, Sigma_det_1, log_p1 = self._cache[key]
            else:
                Gamma_1, Sigma_det_1 = sherman_morrison_update(
                    Gamma, Sigma_det, u, u)
                log_p1 = self.log_p(C, Gamma_1, Sigma_det_1)
                self._cache_insert(key, (Gamma_1, Sigma_det_1, log_p1))

        Z = logsumexp(log_p0, log_p1)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            p0 = np.exp(log_p0 - Z)

        if np.random.rand() > p0:
            # we set it to 1
            C[idx] = 1
            Gamma = Gamma_1
            Sigma_det = Sigma_det_1
            log_pc = log_p1

        else:
            # we set it to 0
            C[idx] = 0
            Gamma = Gamma_0
            Sigma_det = Sigma_det_0
            log_pc = log_p0
        self.likelihoods.append(log_pc)
        return C, Gamma, Sigma_det, log_pc
    
    # @profile
    def get_mvn_params(self, C):
        cache_hit = False
        if not self.use_gibbs:
            for i in range(self.d):
                i_val = C[i]
                C[i] = 1 - i_val
                key = tuple(C.tolist())
                C[i] = i_val
                if key in self._cache:
                    G, s = self._cache[key]
                    u = self.X_td[:, i][:, None]
                    if i_val:
                        v = u
                    else:
                        v = -u
                    Gamma, Sigma_det = sherman_morrison_update(
                        G, s, u, v)
                    cache_hit = True
                    break
        if not cache_hit:    
            Sigma = (self.kappa_sq * self.X_td[:, C>0.5].dot(self.X_td[:, C>0.5].T) + 
                     self.sigma_sq * np.eye(self.X_td.shape[0]))
            # hack to make inference work
            # Sigma = (self.kappa_sq * self.X_td[:, C>0.5].dot(self.X_td[:, C>0.5].T) + 
            #          np.eye(self.X_td.shape[0]))
            Gamma = np.linalg.inv(Sigma)
            Sigma_det = np.linalg.det(Sigma)
        if not self.use_gibbs:
            key = tuple(C.tolist())
            self._cache_insert(key, (Gamma, Sigma_det))
        return Gamma, Sigma_det

    def log_p(self, C, Gamma=None, Sigma_det=None):
        if Gamma is None:
            Gamma, Sigma_det = self.get_mvn_params(C)
        nnz = np.sum(C)            
        return ((self.d - nnz) * np.log(self.eps) + 
                      nnz * np.log(1-self.eps) + 
                      log_mvnpdf(self.Y_t,Gamma, Sigma_det))
    
    def mean(self):
        thetas = self.sample(N=200)
        
        return np.sum(thetas, axis=0) / 200

################################################################################
## auxiliary functions
################################################################################

#@profile
def sherman_morrison_update(A_inv, A_det, u, v):
    """
    Computes the inverse and determinatn of B = A + uv'
    given A^-1 and |A|

    source: 
    https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
    https://en.wikipedia.org/wiki/Matrix_determinant_lemma    
    """
    denom = 1 + v.T.dot(A_inv.dot(u))
    B_inv = A_inv - A_inv.dot(u).dot(v.T.dot(A_inv)) / denom
    B_det = denom * A_det
    # np.linalg.cholesky(B_inv) # to check that B_inv is PSD
    return B_inv, B_det
    

#@profile
def log_mvnpdf(Y, Gamma, Sigma_det, mu=0):
    d = Gamma.shape[0]
    Y = Y - mu
    exponent = -(Y.T.dot(Gamma.dot(Y)))/2
    assert exponent < 0, 'Negative Exponent in MVN'
    assert Sigma_det > 0, 'Non-PSD Mat in MVN'
    return ((-d / 2) * np.log(2*np.pi) + 
            -0.5 * np.log(Sigma_det) + 
            exponent)
    
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

def logsumexp(*args):
    a = np.max(args)
    return a + np.log(np.sum(np.exp(x - a) for x in args))

def get_binary_vals(d, out_arr = None):
    if out_arr is None:
        out_arr = np.zeros((np.power(2, d), d))
    recurse_N = np.power(2, d-1)
    if d > 1:
        get_binary_vals(d-1, out_arr=out_arr[:recurse_N, 1:])
        out_arr[recurse_N:, 1:] = out_arr[:recurse_N, 1:]
    out_arr[:recurse_N, 0] = 0
    out_arr[recurse_N:, 0] = 1
    return out_arr

    
