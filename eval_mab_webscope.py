from __future__ import division

import h5py, sys, copy, time

from IPython import parallel as ipp
from IPython.parallel.util import interactive
from IPython.parallel import require
from IPython.parallel.error import CompositeError, RemoteError

import numpy as np

import mab
from mab import LinMAB, LinBayesUCB, LinParticleGI, LinUCB

from features import MeanFeats

from prob_utils import Beta, Bernoulli, NormalInverseGamma, Gaussian,\
    LinNormal, SpikeSlab

def get_quad_terms(vec):
    N = vec.shape[0]
    v_t_v = np.dot(vec[:, None], vec[None, :])
    inds = np.triu_indices(N)
    return np.r_[vec, v_t_v[inds]]

class WebscopeReader(object):

    DEFAULT_DATA='R6/ydata-fp-td-clicks-v1_0.20090501'

    def __init__(self, datafile=None, line_offset=0):
        if datafile is None:            
            datafile=self.DEFAULT_DATA
        self.f = open(datafile, 'r')
        self.line = line_offset
        for _ in range(line_offset):
            self.f.readline()

    def next(self):
        self.line += 1
        l = self.f.readline().split('|')
        l[-1] = l[-1].strip('\n')
        U = np.zeros((len(l) - 2, 6))
        ts, article_shown, clicked = l[0].strip().split(' ')
        clicked = int(clicked)
        _, user_feats = self.read_feats(l[1])
        articles = []
        article_feats = []
        for x in l[2:]:
            cur_art, cur_feats = self.read_feats(x)
            articles.append(cur_art)
            article_feats.append(np.r_[user_feats, cur_feats, user_feats * cur_feats])

        selected = articles.index(article_shown)
        U = np.array(article_feats)        
        return U, selected, clicked

    def read_feats(self, data):
        data = data.strip()
        data = data.split(' ')
        uid = data[0]
        feats = np.zeros(len(data) - 1)
        for s in data[1:]:
            ind, val = s.split(':')
            feats[int(ind)-1] = float(val)
        return uid, feats

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.f.close()

def timing_eval(policies, N_trials, T):
    result = []
    starts = [(1, 1), (3, 5), (5, 3), (5, 5), (10, 10)]
    for p in policies:
        print 'Testing {}'.format(p.name)
        p.use_cache=False
        start = time.time()
        for n in range(N_trials):
            for s in starts:
                p.index(Beta(s), s[0]+s[1] - 2)
        taken = time.time() - start
        result.append(taken / ( N_trials * len(starts)) )
    return result

def mab_eval(bandit, T, pol_cfg, N_trials=100, seed=None, parallel=False):
    if seed is not None:
        np.random.seed(seed)
        seed += 1

    all_policies = extract_policies(**pol_cfg)
    policies = []
    for p in all_policies:
        if p.name in pol_cfg['names']:
            policies.append(p)
    names = [p.name for p in policies]

    # print [U for (ad, U) in arm_dists]
    results = []
    if parallel == 1:

        rc = ipp.Client(profile='ssh')

        dv = rc[:]
        n_clients = len(dv)
        with dv.sync_imports():
            import mab
        v = rc.load_balanced_view()
    
        print 'Evaluating Policies {}'.format(names)
        
        results = v.map(eval_helper, [bandit.theta_prior] * N_trials, 
                        [T]*N_trials, [pol_cfg]*N_trials, [frozenset(names)] * N_trials,
                        [(seed + inum) for inum in range(N_trials)])

        start = time.time()
        rate = 0
        n_complete = 0

        while rc.outstanding:
            try:
                rc.wait(rc.outstanding, 1e-1)
            except ipp.TimeoutError:
                # ignore timeouterrors
                pass
            if n_complete < N_trials - len(rc.outstanding):
                n_complete = N_trials - len(rc.outstanding)
                rate = ((time.time() - start) / n_complete)
            if n_complete > 0:
                est_remaining = rate * len(rc.outstanding)
            else:
                est_remaining = 'No Estimate'
            sys.stdout.write(
                '\rFinished {} / {} jobs\tEstimated Time Remaining: {:.4}'.format(
                    n_complete, N_trials, est_remaining))
            sys.stdout.flush()
    elif parallel == 2:
        from joblib import Parallel, delayed
        print 'Evaluating Policies {}'.format(names)
        results = Parallel(n_jobs=7, verbose=50)(delayed(_eval_helper)(
                bandit.theta_prior, T, pol_cfg, names, seed + inum) for 
                inum in range(N_trials))
    else:
        for inum in range(N_trials):
            results.append(
                eval_helper(
                    bandit.theta_prior, T, pol_cfg, names, seed=seed+inum))
            sys.stdout.write("{} / {}\t".format(inum, N_trials))
            sys.stdout.flush()
    means = []
    variances = []
    avg_err = []
    discounted_mean = []
    try:
        if type(results[0]) == list:
            results = [x[0] for x in results]
    except CompositeError, e:
        print e
        import IPython; IPython.embed()
        
    for j in range(len(policies)):
        try:
            regrets, choices, discounted = results[0].get()
        except CompositeError, e:
            print e
            import IPython; IPython.embed()
        regrets = regrets[j]
        choices = choices[j]
        discounted = discounted[j]
        errors = np.array(choices != bandit.ibest, dtype=np.int)
        for i in range(1, N_trials):
            regrets_i, choices, discounted_i = results[i].get()
            regrets = np.c_[regrets, regrets_i[j]]
            errors += (choices[j] != bandit.ibest)
            discounted += discounted_i[j]
        discounted /= N_trials
        discounted_mean.append(discounted)
        means.append(np.mean(regrets, axis=1))
        variances.append(np.var(regrets, axis=1))
        avg_err.append(errors / N_trials)
    return means, variances, avg_err, discounted_mean

def create_append(mean_g, p, l, run_k, overwrite):
    name = p.name
    if name not in mean_g:
        mean_g.create_group(name)
    if overwrite and run_k in mean_g[p.name]:
        del mean_g[p.name][run_k]
    if run_k not in mean_g[p.name] and p not in l:
        l.append(p)
    

def extract_policies(pgi_N, pgi_T, gamma, bayesucb, 
                     thompson, ucb, H, use_cache, 
                     run_k, run_type, overwrite, names=None):
    policies = []
    if use_cache:        
        f = h5py.File('.lin_results.{}.h5'.format(run_type), 'a')
        mean_g = f['mean']
        discounted_g = f['discounted_reward']

    if bayesucb:
        p = LinBayesUCB()
        if use_cache:
            create_append(mean_g, p, policies, run_k, overwrite)
            create_append(discounted_g, p, policies, run_k, overwrite)
        else:
            policies.append(p)

    if thompson:
        p = LinParticleGI(1, 1, 1, MeanFeats())
        p.name = 'Thompson Sampling'
        if use_cache:
            create_append(mean_g, p, policies, run_k, overwrite)
            create_append(discounted_g, p, policies, run_k, overwrite)
        else:
            policies.append(p)

    if ucb:
        p = LinUCB()
        if use_cache:
            create_append(mean_g, p, policies, run_k, overwrite)
            create_append(discounted_g, p, policies, run_k, overwrite)
        else:
            policies.append(p)

    for g in gamma:
        for n in pgi_N:
            for t in pgi_T:
                p = LinParticleGI(n, g, t, MeanFeats())
                if use_cache:
                    create_append(mean_g, p, policies, run_k, overwrite)
                    create_append(discounted_g, p, policies, run_k, overwrite)
                else:
                    policies.append(p)

    if use_cache:
        f.close()
    return policies

def _eval_helper(arm_dists, arm_prior, T, pol_cfg, names, seed=None):
    return eval_helper(arm_dists, arm_prior, T, pol_cfg, names, seed)



# @profile
@interactive
@require('mab', 'eval_lin_mab', 'prob_utils')
def eval_helper(prior, T, pol_cfg, names, seed=None):
    import mab
    reload(mab)
    from mab import LinMAB, LinBayesUCB, LinParticleGI, LinUCB
    import eval_mab_webscope
    reload(eval_mab_webscope)
    from eval_mab_webscope import extract_policies, EvalResult
    import numpy as np
    import copy

    all_policies = extract_policies(**pol_cfg)
    policies = []
    for p in all_policies:
        if p.name in names:
            policies.append(p)

    if seed:
        np.random.seed(seed)
        dist_seed = int(np.random.rand()*1e16)
        pol_seeds = dict([(p, int(np.random.rand()*1e16)) for p in policies])
        np.random.seed(dist_seed)
    # so that tie-breaking 
    # is independent for policies
    # np.random.seed()
    rewards = []
    # discounted_rewards = []
    gamma = 0.9    
    discount_seq = [gamma**t for t in range(T)]
    for pol in policies:
        with WebscopeReader() as reader:
            print 'Evaluating policy:\t{}'.format(pol.name)
            np.random.seed(pol_seeds[pol])
            bel = copy.deepcopy(prior)
            r = np.zeros(T)
            for t in xrange(T):
                print t
                pol_i = -1
                data_i = -2
                while pol_i != data_i:
                    bel.U, data_i, r_t = reader.next()
                    pol_i = pol.select_best(bel, t)
                r[t] = r_t
                bel = bel.update((r[t], bel.U[pol_i]))
            rewards.append(r)
    rewards = np.array(rewards)

    return EvalResult(rewards)

class EvalResult(object):

    def __init__(self, rewards):
        self.rewards = rewards[:]
        self.rewards.flags.writeable=False

    def get(self):
        return (self.rewards,) 
        
    def __hash__(self):
        self.rewards.flags.writeable=False
        return hash((self.rewards.data,))

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, choices=['normal', 'spike-slab'])
    parser.add_argument('n_trials', type=int)
    parser.add_argument('T', type=int, default=20)
    parser.add_argument('--pgi_N', type=int, nargs='*', default=[100])
    parser.add_argument('--pgi_T', type=int, nargs='*', default=[50])
    parser.add_argument('--gamma', type=float, nargs='*', default=[0.9])
    parser.add_argument('--bayesucb', action='store_true')
    parser.add_argument('--ucb', action='store_true')
    parser.add_argument('--thompson', action='store_true')
    parser.add_argument('--timing_analysis', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--parallel', type=int, default=0)
    parser.add_argument('--nnz', type=int, default=2)
    parser.add_argument('--reseed', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    outf = h5py.File('.webscope_results.{}.h5'.format(args.type), 'a')
    run_k = str((args.n_trials, args.T, args.nnz))

    kappa_sq = 10
    sigma_sq = 1
    eps = 0.8

    D = 18
    
    if args.type == 'normal':
        theta_prior = LinNormal(kappa_sq, sigma_sq, D)
        noise_dist  = Gaussian(0, np.sqrt(sigma_sq))
    elif args.type == 'spike-slab':
        sparsity_pattern = np.zeros(D)
        sparsity_pattern[:args.nnz] = 1
        theta_prior = SpikeSlab(eps, kappa_sq, sigma_sq, D,
                                sparsity_pattern=sparsity_pattern)
        print 'Using Gibbs:\t{}'.format(theta_prior.use_gibbs)      
        noise_dist  = Gaussian(0, np.sqrt(sigma_sq))
    else:
        raise NotImplemented

    if args.reseed or 'seed' not in outf:
        for g in outf:
            del outf[g]
        outf['seed'] = int(time.time())

    seed = outf['seed'][()]

    mean_g = outf.create_group('mean') if 'mean' not in outf else outf['mean']
    time_g = outf.create_group('time') if 'time' not in outf else outf['time']
    discount_g = outf.create_group('discounted_reward') if 'discounted_reward' \
        not in outf else outf['discounted_reward']
    theta_g = outf.create_group('theta') if 'theta' not in outf else outf['theta']

    if run_k not in theta_g:
        np.random.seed(seed)
        theta_g[run_k] = theta_prior.sample_init().T

    m = LinMAB(theta_prior, noise_dist, 
               2, D, theta=theta_g[run_k][:])
    
    pol_cfg = {'pgi_N': args.pgi_N,
               'pgi_T': args.pgi_T,
               'gamma': args.gamma,
               'bayesucb': args.bayesucb,
               'thompson': args.thompson,
               'ucb': args.ucb,
               'H': args.T,
               'use_cache': not args.timing_analysis,
               'overwrite': args.overwrite,
               'run_k': run_k,
               'run_type': args.type}

    policies = extract_policies(**pol_cfg)    
    params_list = [p.name for p in policies]

    pol_cfg['names'] = params_list
    pol_cfg['use_cache'] = False

    if args.timing_analysis:
        times = timing_eval(policies, args.n_trials, args.T)
        for i, p in enumerate(policies):            
            print "{} : {}".format(p.name, times[i])
            if p.name not in time_g:
                pg = time_g.create_group(p.name)
            if run_k in pg:
                del pg[run_k]
            pg[run_k] = times[i]
        import sys; sys.exit(0)

    mean, variance, err_rate, discounted = mab_eval(
        m, args.T, pol_cfg, N_trials=args.n_trials, seed=(seed+1), parallel=args.parallel)
    
    print "Regret"
    for i, k in enumerate(params_list):
        print "{} mean: ".format(k), mean[i]
        mean_g[k][run_k] = mean[i]

    print "Discounted"
    for i, k in enumerate(params_list):
        print "{} mean: ".format(k), discounted[i]
        discount_g[k][run_k] = discounted[i]

