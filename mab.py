from __future__ import division

import numpy as np
import scipy.stats as ss

import copy, sys, h5py, time
import IPython as ipy

from solvers import gi_outer_loop, bernoulli_ret_val, fh_bernoulli_ret_val,\
    ParticleRetALP, ParticleRetALPv2, SVMRetVal

DEBUG = True
eps = 1e-4

################################################################################
##    Policies
################################################################################

class IndexPolicy(object):
    
    def __init__(self):
        self._cache = {}
        self.use_cache = False

    def index(self, prior, H):
        if self.use_cache:
            if prior in self._cache:
                return self._cache[prior]
        ind = self._index(prior, H)
        if self.use_cache:
            self._cache[prior] = ind
        return ind

    def select_best(self, priors, H):
        inds = np.array([self.index(p, H) for p in priors])
        besti = np.nonzero(inds == np.max(inds))[0]
        return np.random.choice(besti)

class BayesUCB(IndexPolicy):

    def __init__(self):
        super(BayesUCB, self).__init__()
        self.name = 'Bayes-UCB'
        try:
            self.cdf(0)
            self.exact_quantile=True
        except AttributeError:
            self.exact_quantile=False
    
    def _index(self, prior, H):
        return prior.quantile(1 - 1/(H+1), exact=self.exact_quantile)

class UCB(IndexPolicy):

    def __init__(self):
        super(UCB, self).__init__()
        self.name = 'UCB'
    
    def _index(self, prior, H):
        (a, b) = prior.alpha, prior.beta
        return prior.mean() + np.sqrt(2 * np.log(H+1) / (a + b))

class ParticleGI(IndexPolicy):
    
    def __init__(self, N, gamma, T, feats, use_alp=False):
        super(ParticleGI, self).__init__()
        self.N = N
        self.T = T
        self.use_cache = False
        self.name = "Particle GI {},{}".format((N, T), gamma)
        if use_alp:
            self.solver = ParticleRetALPv2(gamma, feats, T, N)
        else:
            self.solver = SVMRetVal(gamma, feats, T, N)
        # maps a history of rewards to a vector
        # of features

    def _index(self, prior, H):
        N, T = self.N, self.T
        np.random.seed()
        thetas = prior.sample(N)
        R = np.zeros((N, T))
        nu = 0
        for n, theta in enumerate(thetas):
            mu_n = theta.mean()
            R_n = theta.sample(T)
            self.solver.set_particle(n, mu_n, R_n, update=(n==N-1) )
        start = time.time()
        nu = self.solver.solve()
        return nu

class ParticleFHGI(ParticleGI):

    def __init__(self, N, T, FH, feats, use_alp=False):
        super(ParticleFHGI, self).__init__(N, 1, T, feats)
        self.FH = FH
        self.solver = SVMRetVal(1, feats, T, N, FH=FH)
        self.name = "Particle FHGI {}".format((self.N, self.T))

    def _index(self, prior, H):
        EFH = self.FH - H
        if EFH < 0: return 0
        self.solver.FH = EFH
        return super(ParticleFHGI, self)._index(prior, 0)

class ThompsonSampling(ParticleGI):
    
    def __init__(self):
        super(ThompsonSampling, self).__init__(1, 0.9, 1, lambda x: x)        
        self.use_cache = False
        self.name = "Thompson Sampling"

class BernoulliGI(IndexPolicy):
    
    def __init__(self, gamma, tol = eps, fcache=True):
        super(BernoulliGI, self).__init__()
        self.gamma = gamma
        self.tol = tol
        self.use_cache = True
        self.name = "IH GI {}".format(self.gamma)
        # if fcache:
        #     self._fcache = h5py.File('/home/dhm/Dropbox/cs294_fa14_project/2armed/.bernoulligi.cache.ih', 'r')
        # else:
        #     self._fcache=None


    def _index(self, prior, H):
        ab = (prior.alpha, prior.beta)
        # if self._fcache is not None:
        #     return self._fcache[str(self.gamma)][str(ab)][()]
        rv_fn = lambda r: bernoulli_ret_val(r, self.gamma, 
                                            ab, self.tol)
        res = gi_outer_loop(rv_fn)
        return res * (1-self.gamma)

class BernoulliFHGI(BernoulliGI):
    
    def __init__(self, T, tol = eps, fcache=True):
        super(BernoulliFHGI, self).__init__(1, tol)
        self.T = T
        self.use_cache = False
        self.name = "FHGI"
        # if fcache:
        #     self._fcache = h5py.File('/home/dhm/Dropbox/cs294_fa14_project/2armed/.bernoulligi.cache.fh', 'r')
        # else:
        #     self._fcache=None

    def _index(self, prior, H):
        ab = (prior.alpha, prior.beta)
        # if self._fcache is not None:
        #     return self._fcache[str(self.T)][str(ab)][()]
        rv_fn = lambda r: fh_bernoulli_ret_val(r, self.T - H,
                                               ab, self.tol)
        res = gi_outer_loop(rv_fn)
        return res / (self.T - H)
        
class MAB(object):

    def __init__(self, arm_prior, k, arm_init=None):
        self.arm_prior = arm_prior
        self.k = k
        if arm_init:
            self.arm_dists = arm_init
            self.update_arms()
            self.resample=False
        else:
            self.resample=True
            self.resample_arms()

    def resample_arms(self):
        if self.resample:
            self.arm_dists = self.arm_prior.sample(self.k)
            self.update_arms()
        return copy.deepcopy(self.arm_dists)

    def update_arms(self):
        self.arm_means = [d.mean() for d in self.arm_dists]
        self.ibest = np.argmax(self.arm_means)
