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
    
    def feature(self, history):
        return np.mean(history)

    @staticmethod
    def get_size():
        return 1

class MeanVarFeats(Feature):
    
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
