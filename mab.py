from __future__ import division

import numpy as np
import scipy.stats as ss

import copy, sys, h5py, time
import IPython as ipy

from solvers import gi_outer_loop, bernoulli_ret_val, fh_bernoulli_ret_val,\
    ParticleRetALP, ParticleRetALPv2, SVMRetVal, SVMRetValv2

from prob_utils import DiracDelta, SumDist

from features import MeanVarFeats, MeanFeats

DEBUG = True
eps = 1e-4

def rand_besti(arr):
    besti = np.nonzero(arr == np.max(arr))[0]
    return np.random.choice(besti)

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
        return rand_besti(
            np.array([self.index(p, H) for p in priors]))

class LinIndexPolicy(IndexPolicy):
    
    def index(self, prior, H, j):
        return self._index(prior, H, j)

    def select_best(self, prior, H):
        self.thetas = prior.sample(self.N)
        return rand_besti(
            np.array([self.index(prior, H, j) for j in range(prior.U.shape[0])]))

class BayesUCB(IndexPolicy):

    def __init__(self):
        super(BayesUCB, self).__init__()
        self.name = 'Bayes-UCB'
    
    def _index(self, prior, H):
        return prior.quantile(1 - 1/(H+1))

class LinBayesUCB(LinIndexPolicy):

    N = 100
    name = 'Bayes-UCB'
    
    def __init__(self):
        super(LinBayesUCB, self).__init__()
        self.thetas = None
    # @profile
    def _index(self, prior, H, j):
        v = 1 - 1/(H+1)
        means = sorted(self.thetas.dot(prior.U[j, :]))
        min_q = int(np.floor(v * self.N))
        try:
            return (means[min_q] + means[min_q+1]) / 2
        except IndexError:
            return means[min_q]

class UCB(IndexPolicy):

    def __init__(self):
        super(UCB, self).__init__()
        self.name = 'UCB1'
    
    def _index(self, prior, H):
        (a, b) = prior.alpha, prior.beta
        return prior.mean() + np.sqrt(2 * np.log(H+1) / (a + b))

class Bootstrap(IndexPolicy):
    """
    a policy that uses the rate of rewards of UCB
    from a restart-in-state formulation as an index
    """
    N = 300
    T = 50
    def __init__(self, base_pol):
        super(Bootstrap, self).__init__()
        self.base = base_pol
        self.name = 'i-{}'.format(self.base.name)
        
    def _index(self, prior, H):
        thetas = prior.sample(self.N)
        mu = np.array([theta.mean() for theta in thetas])
        times = np.zeros(self.N)
        for i, theta in enumerate(thetas):
            prior_theta = prior
            H_theta = H
            R_theta = theta.sample(self.T)
            for t in range(self.T):
                prior_theta = prior_theta.update(R_theta[t])                
                if self.base.select_best([prior_theta, prior], H_theta) != 0:
                    times[i] = t+1
                    break
        avg_reward = np.mean(mu * times)
        avg_time   = np.mean(times)
        return avg_reward / avg_time

class UCBN(IndexPolicy):
    
    def __init__(self):
        super(UCBN, self).__init__()
        self.name = 'UCB1-Norm'

    def _index(self, prior, H):
        # import pdb; pdb.set_trace()
        n = prior.nu
        # if H == 0 or n <= np.ceil(np.log(H)):
        #     return np.inf
        if H == 0 or n < 2:
            return np.inf

        mu = prior.mean()
        ucb = np.sqrt(16 * (prior.q - n * np.power(mu, 2)) / (n - 1) *
                      np.log(H-1) / n)
        if np.isnan(ucb):
            ucb = np.inf
        return mu + ucb

class LinUCB(LinIndexPolicy):

    N = 1

    def __init__(self):
        super(LinUCB, self).__init__()
        self.name = 'Lin-UCB'
        self.thetas = None

    def _index(self, prior, H, j):
        if prior.X_td is None or prior.X_td.shape[1] > prior.X_td.shape[0]:            
            return np.inf
        # precision = np.linalg.inv(prior.X_td.T.dot(prior.X_td))
        ucb = prior.U[j, :].T.dot(prior.cov.dot(prior.U[j, :]))
        return prior.mean().dot(prior.U[j, :]) + np.sqrt(np.log(H+1)) * ucb
        

class ParticleGI(IndexPolicy):
    
    def __init__(self, N, gamma, T, feats, v2=False):
        super(ParticleGI, self).__init__()
        self.N = N
        self.T = T
        self.use_cache = False
        self.name = "Particle GI {},{}".format((N, T), gamma)
        # if v2:
        #     self.solver = SVMRetValv2(gamma, feats, T, N)
        #     self.name+='v2'
        # else:
        self.solver = SVMRetValv2(gamma, feats, T, N)
        # feats maps a history of rewards to a vector
        # of features

    def _index(self, prior, H):
        N, T = self.N, self.T
        thetas = prior.sample(N)
        R = np.zeros((N, T))
        nu = 0
        for n, theta in enumerate(thetas):
            mu_n = theta.mean()
            R_n = theta.sample(T)
            self.solver.set_particle(n, mu_n, R_n, update=(n==N-1) )
        nu = self.solver.solve()
        return nu

        

class LinParticleGI(LinIndexPolicy):

    def __init__(self, N, gamma, T, feats):
        super(LinParticleGI, self).__init__()
        self.N = N
        self.T = T
        self.use_cache = False
        self.name = "Particle GI {},{}".format((N, T), gamma)
        self.gamma = gamma
        self.feats = feats
        self.solver = SVMRetValv2(gamma, feats, T, N)

    # @profile
    def _index(self, prior, H, j):
        N, T = self.N, self.T
        thetas = prior.sample(N, prior.U[j, :])
        R = np.zeros((N, T))
        nu = 0
        for n, theta in enumerate(thetas):
            mu_n = theta.mean()
            R_n = theta.sample(T)
            self.solver.set_particle(n, mu_n, R_n, update=(n==N-1) )
        nu = self.solver.solve()
        return nu

    # @profile
    def select_best(self, prior, H):
        K = prior.U.shape[0]
        self.thetas = prior.sample(self.N)


        eps = np.random.normal(0, np.sqrt(prior.sigma_sq), (self.N, self.T))
        mus = self.thetas.dot(prior.U.T)
        ubs = mus.max(axis=0)
        lbs = mus.mean(axis=0)
        alive = np.ones(K)
        besti = np.argmax(lbs)
        bestlb = lbs[besti]
        solvers = [SVMRetValv2(self.gamma, self.feats, self.T, self.N)
                   for k in range(K)]
        for k in range(K):            
            for i in range(self.N):
                solvers[k].set_particle(i, mus[i, k], mus[i, k] + eps[i, :])
        while np.sum(alive) > 1:
            lmbda = solvers[besti].solve()
            ubs[besti] = lmbda
            lbs[besti] = lmbda
            for k in range(K):
                if not alive[k]:
                    continue
                rv = solvers[k].ret_val(lmbda)
                if rv > lmbda:
                    lbs[k] = rv
                else:
                    ubs[k] = lmbda
            besti = np.argmax(lbs)
            bestlb = lbs[besti]
            alive = ubs > bestlb
        return besti

        # old_besti = rand_besti(
        #     np.array([self.index(prior, H, j) for j in range(prior.U.shape[0])]))
        # print "Old Running Time:\t{}".format(time.time() - start)
        # return besti

class ThompsonSampling(ParticleGI):
    
    def __init__(self):
        super(ThompsonSampling, self).__init__(1, 1, 1, MeanVarFeats())
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

class LinMAB(MAB):

    def __init__(self, theta_prior, noise_dist, k, d, theta=None):
        self.k = k
        self.d = d
        self.theta_prior = theta_prior
        self.noise_dist = noise_dist
        if theta is None:
            self.theta = theta_prior.sample_init().T
        else:
            self.theta = theta

        self.resample_arms()

    def resample_arms(self):
        # generate a random set of 
        # vectors on the unit sphere
        self.U = np.random.rand(self.k, 
                                self.d)
        self.U = (self.U / 
                  self.U.sum(axis=1)[:, None])
        self.arm_dists = []
        self.arm_means = self.U.dot(self.theta)
        for i in range(self.k):
            d_i = SumDist(DiracDelta(self.arm_means[i]), 
                          self.noise_dist)
            d_i.U = self.U[i, :]
            self.arm_dists.append(d_i)
        self.update_arms()
        return copy.deepcopy(self.arm_dists), self.U
