from __future__ import division

from mab import ParticleGI, ParticleFHGI, BernoulliGI, BernoulliFHGI
from prob_utils import Bernoulli, Beta
from features import MeanVarFeats

import numpy as np
import matplotlib.pyplot as plt
import h5py, argparse, os
from joblib import Parallel, delayed

import time

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fh', action='store_true')
    parser.add_argument('--ih', action='store_true')
    return parser.parse_args()
args = parse_arguments()

ih_test_states = [Beta(s) for s in [(1, 1), (3, 5)]] 

N_particles = range(1, 150)
N_particles = [1, 3, 10, 30, 100]#, 300]#, 1000]
# N_particles = [100, 300]
# N_particles=[300]
T_vals = [5, 10, 50, 100]#, 300]
# T_vals = [100]#, 300]

fh_test_states =[Beta(s) for s in [(1, 1), (3, 5)]]
H_vals = [10, 100, 300]

ihp_gt = BernoulliGI(0.9, fcache=False)
fhp_gt = dict([(h, BernoulliFHGI(h, fcache=False)) for h in H_vals])
ihgt = {}
fhgt = {}
for prior in ih_test_states:
    ihgt[(prior.alpha, prior.beta)] = ihp_gt.index(prior, 0)

for prior in fh_test_states:
    for h in H_vals:
        t = prior.alpha + prior.beta - 2        
        fhgt[(prior.alpha, prior.beta, h)] = fhp_gt[h].index(
            prior, 0)

ihresults = {}
fhresults = {}

N_samples = 100

from joblib import Parallel, delayed

def ihpgi(n, g, t, f, prior, p=None):
    if p is None:
        p = ParticleGI(n, g, t, f)
    r = p.index(Beta(prior), 0)
    return r

def fhpgi(n, h, t, f, (a, b), p=None):
    if p is None:
        p = ParticleFHGI(n, t, h, f)
    res = p.index(Beta((a, b)), 0)
    return res



for n in N_particles:
    for t in T_vals:        
        if args.ih:
            print 'testing {}, {}'.format(n, t)
            for s in ih_test_states:
                a, b = s.alpha, s.beta
                start = time.time()
                estimates = Parallel(n_jobs=7, verbose=0)(
                    delayed(ihpgi)(n, 0.9, t, MeanVarFeats(), (a, b))
                    for _ in xrange(N_samples))
                # p = ParticleGI(n, 0.9, t, MeanVarFeats())
                # # p.solver.model.setParam('OutputFlag', True)
                # start = time.time()
                # estimates = [p.index(Beta((a, b)), 0) for _ in xrange(N_samples)]
                estimates = np.asarray(estimates)
                error = estimates - ihgt[(a, b)]
                ihresults[(s, n, t)] = np.mean(np.power(error, 2))
                print "mse: {}, avg err: {}, time/solve: {}".format(
                    np.mean(np.power(error, 2)), np.mean(error), (time.time() - start)/N_samples)
                
        if args.fh:
            print 'testing {}'.format(n)
            for s in fh_test_states:
                a, b = s.alpha, s.beta
                for h in H_vals:
                    print 'testing {}, {}, {}, {}'.format(n, h, t, (s.alpha, s.beta))
                    start = time.time()
                    estimates = Parallel(n_jobs=7, verbose=0)(
                        delayed(fhpgi)(n, h, t, MeanVarFeats(), (a, b))
                        for _ in xrange(N_samples))
                    estimates = np.asarray(estimates)
                    error = (estimates - fhgt[(a, b, h)])
                    fhresults[(h, s, n, t)] = np.mean(np.power(error, 2))
                    print "mse: {}, avg err: {}, time/solve: {}".format(
                        np.mean(np.power(error, 2)), np.mean(error), (time.time() - start)/N_samples)

if args.ih:
    f, axarr = plt.subplots(len(ih_test_states), sharex=True)

    for i, s in enumerate(ih_test_states):
        for T in T_vals:
            res_t = []
            for n in N_particles:
                res_t.append(ihresults[(s, n, T)])
            axarr[i].plot(N_particles, res_t, lw=2.0, label='Lookahead {}'.format(T))
            axarr[i].legend(loc='best')
            axarr[i].set_title("({}, {})".format(s.alpha, s.beta))
            axarr[i].set_ylabel('Mean Squared Error')
    plt.xlabel('Number of Samples')
    plt.suptitle("Error in Infinite Horizeon Particle GI vs Number of Samples")

    plt.savefig("figs/mse_ih.pdf")

if args.fh:
    for i, h in enumerate(H_vals):
        f, axarr = plt.subplots(len(fh_test_states), sharex=True)
        for j, s in enumerate(fh_test_states):
            for T in T_vals:
                res_t = []
                for n in N_particles:
                    res_t.append(fhresults[(h, s, n, T)])
                axarr[j].plot(N_particles, res_t, lw=2.0, label='Lookahead {}'.format(T))
                axarr[j].legend(loc='best')
            axarr[j].set_ylabel('Mean Squared Error')
            axarr[j].set_title("({}, {})".format(s.alpha, s.beta))
            axarr[j].legend(loc='best')
        plt.xlabel('Number of Samples')

        plt.suptitle("Mean Squared Error in Particle AP Calculations with H={}".format(h))
        plt.savefig("figs/mse_fh_{}.pdf".format(h))

