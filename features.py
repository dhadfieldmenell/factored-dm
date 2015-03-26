"""
Functions and classes for computing features
"""
from __future__ import division

import numpy as np

class Feature(object):
    """
    base class for computing features
    """

    def feature(self, history, **kwargs):
        """
        returns the feature for this state/segname
        """
        raise NotImplemented
        
    @staticmethod
    def get_size():
        raise NotImplemented

class MeanFeats(Feature):
    
    def all_features(self, history):
        N = history.shape[0]
        phi = np.cumsum(history) / np.linspace(1, N, N)
        return phi[1:, None]

    def feature(self, history):
        return np.mean(history)

    @staticmethod
    def get_size():
        return 1

class MeanVarFeats(Feature):

    def all_features(self, history):
        N = history.shape[0]
        phi = np.zeros((N-1, 2))
        means = np.cumsum(history) / np.linspace(
            1, N, N, dtype=np.int)
        for i in range(N-1):
            phi[i, 0] = means[i+1]
            phi[i, 1] = np.linalg.norm(history[:i+1] - phi[i, 0])
        return phi
    
    def feature(self, history):
        return np.r_[np.mean(history), np.var(history)]

    @staticmethod
    def get_size():
        return 2

class QuadFeats(Feature):

    def __init__(self, linfeats):
        self.linfeats = linfeats
    
    def feature(self, history):
        linear = self.linfeats(history)
        l = linear.get_size()
        quad = []
        for i in range(l):
            for j in range(i+1):
                quad.append(linear[i]*linear[j])
        quad.extend(linear)
        return np.array(linear)

    @staticmethod
    def get_size():
        """
        l = linfeats.get_size()
        [linear, cross-terms, quad]
          l       l(l-1)/2     l
        """
        l = self.linfeats.get_size()
        return 2*l + (l *(l-1)) / 2

class MeanRhoFeats(Feature):
    
    def all_features(self, history, lmbda):
        N = history.shape[0]
        phi = np.cumsum(history) / np.linspace(1, N, N)
        return phi[1:, None] - lmbda

    def feature(self, history):
        return np.mean(history)

    @staticmethod
    def get_size():
        return 1


class NMarkovFeats(Feature):

    def __init__(self, N):
        self.N = N

    def get_size(self):
        return self.N + 3

    def all_features(self, history, Q_ret):
        N, T = history.shape
        M = self.get_size()
        phi = np.zeros((N, M, T))
        for t in xrange(1, T):
            phi[:, :, t] = self.feature(history[:, :t], Q_ret[t])
        return phi

    def feature(self, history, rho=0):
        mu = np.mean(history, axis=1)
        N_data, T = history.shape
        prev_N = np.ones((N_data, self.N)) * mu[:, None]
        prev_N[:, -T:] = history[:, -self.N:]
        ones = np.ones(history.shape[0])
        rho_col = ones * rho
        T_col = T * ones
        feats = np.c_[prev_N, mu, rho_col, T_col]
        return feats
        # feats_norm = feats / feats.sum(axis=1)[:, None]
        # feats_norm[np.isnan(feats_norm)] = 0
        # return feats_norm
        
