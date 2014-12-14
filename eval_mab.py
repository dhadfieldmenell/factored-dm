import h5py, sys, copy, time

from IPython import parallel as ipp
from IPython.parallel.util import interactive
from IPython.parallel import require

import numpy as np

import mab
from mab import MAB, BernoulliGI, BayesUCB, ThompsonSampling,\
    ParticleGI, ParticleFHGI, BernoulliFHGI, UCB

from features import MeanVarFeats

from prob_utils import Beta, Bernoulli, NormalInverseGamma, Gaussian

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

    all_policies = extract_policies(**pol_cfg)
    policies = []
    for p in all_policies:
        if p.name in pol_cfg['names']:
            policies.append(p)
    names = [p.name for p in policies]

    arm_dists = [bandit.resample_arms() for _ in range(N_trials)]
    results = []
    if parallel == 1:                   

        rc = ipp.Client(profile='ssh')

        dv = rc[:]
        n_clients = len(dv)
        with dv.sync_imports():
            import mab
        v = rc.load_balanced_view()
    
        print 'Evaluating Policies {}'.format(names)
        
        results = v.map(eval_helper, arm_dists, [bandit.arm_prior] * N_trials, 
                        [T]*N_trials, [pol_cfg]*N_trials, [frozenset(names)] * N_trials,
                        [seed + inum for inum in range(N_trials)])

        start = time.time()        
        while rc.outstanding:
            try:
                rc.wait(rc.outstanding, 1e-1)
            except ipp.TimeoutError:
                # ignore timeouterrors
                pass
            n_complete = N_trials - len(rc.outstanding)
            if n_complete > 0:
                est_remaining = ((time.time() - start) / n_complete) * len(rc.outstanding)
            else:
                est_remaining = 'No Estimate'
            sys.stdout.write('\rFinished {} / {} jobs\tEstimated Time Remaining: {}'.format(n_complete, N_trials, est_remaining))
            sys.stdout.flush()
    elif parallel == 2:
        from joblib import Parallel, delayed
        print 'Evaluating Policies {}'.format(names)
        results = Parallel(n_jobs=7, verbose=50)(delayed(_eval_helper)(
                ad, bandit.arm_prior, T, pol_cfg, names, seed + inum) for 
                inum, ad in enumerate(arm_dists))
    else:
        for inum, ad in enumerate(arm_dists):
            results.append(eval_helper(ad, bandit.arm_prior, T, pol_cfg, names, seed=seed+inum))
            sys.stdout.write("{} / {}\t".format(inum, N_trials))
            sys.stdout.flush()
    means = []
    variances = []
    avg_err = []
    discounted_mean = []
    for j in range(len(policies)):
        regrets, choices, discounted = results[0].get()
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

def create_append(mean_g, p, l, run_k):
    name = p.name
    if name not in mean_g:
        mean_g.create_group(name)
    if run_k not in mean_g[p.name] and p not in l:
        l.append(p)
    

def extract_policies(pgi_N, pgi_T, fhgi, pfhgi, gi_gamma, gi, ucb,
                     bayesucb, thompson, pbayesucb, H, use_cache, 
                     run_k, run_type, names=None):
    policies = []
    if use_cache:        
        f = h5py.File('.results.{}.h5'.format(run_type), 'a')
        mean_g = f['mean']
        discounted_g = f['discounted_reward']
    if gi:
        p = BernoulliGI(gi_gamma[0])        
        if use_cache:
            create_append(mean_g, p, policies, run_k)
            create_append(discounted_g, p, policies, run_k)
        else:
            policies.append(p)

    if fhgi:
        p = BernoulliFHGI(H)
        if use_cache:
            create_append(mean_g, p, policies, run_k)
            create_append(discounted_g, p, policies, run_k)
        else:
            policies.append(p)
    if ucb:
        p = UCB()
        if use_cache:
            create_append(mean_g, p, policies, run_k)
            create_append(discounted_g, p, policies, run_k)
        else:
            policies.append(p)

    if bayesucb:
        p = BayesUCB()
        if use_cache:
            create_append(mean_g, p, policies, run_k)
            create_append(discounted_g, p, policies, run_k)
        else:
            policies.append(p)
    if thompson:
        p = ThompsonSampling()
        if use_cache:
            create_append(mean_g, p, policies, run_k)
            create_append(discounted_g, p, policies, run_k)
        else:
            policies.append(p)
    for g in gi_gamma:
        for n in pgi_N:
            for t in pgi_T:
                p = ParticleGI(n, g, t, MeanVarFeats())
                if use_cache:
                    create_append(mean_g, p, policies, run_k)
                    create_append(discounted_g, p, policies, run_k)
                else:
                    policies.append(p)
    if pfhgi:
        for n in pgi_N:
            for t in pgi_T:
                p = ParticleFHGI(n, t, H, MeanVarFeats())
                if use_cache:
                    create_append(mean_g, p, policies, run_k)
                else:
                    policies.append(p)

    if use_cache:
        f.close()
    return policies

def _eval_helper(arm_dists, arm_prior, T, pol_cfg, names, seed=None):
    return eval_helper(arm_dists, arm_prior, T, pol_cfg, names, seed)

@interactive
@require('mab', 'eval_mab', 'prob_utils')
def eval_helper(arm_dists, arm_prior, T, pol_cfg, names, seed=None):
    import mab
    reload(mab)
    from mab import MAB, BernoulliGI, BayesUCB, ThompsonSampling,\
        ParticleGI, ParticleFHGI, BernoulliFHGI, UCB
    import eval_mab
    reload(eval_mab)
    from eval_mab import extract_policies, EvalResult
    import numpy as np
    all_policies = extract_policies(**pol_cfg)
    policies = []
    for p in all_policies:
        if p.name in names:
            policies.append(p)

    k = len(arm_dists)
    means = np.array([d.mean() for d in arm_dists])
    ibest = np.random.choice(np.nonzero(means == np.max(means))[0])
    R = np.zeros((k, T))
    if seed:
        np.random.seed(seed)
    for i, d in enumerate(arm_dists):
        R[i, :] = d.sample(T)
    r_best = R[ibest, :]
    # so that tie-breaking 
    # is independent for policies
    np.random.seed()
    regrets = []
    choices = []
    discounted_rewards = []
    gamma = 0.9    
    discount_seq = [gamma**t for t in range(T)]
    for pol in policies:
        bels = [arm_prior for _ in range(k)]
        i = np.zeros(T, np.int)
        r = np.zeros(T)
        for t in xrange(T):
            i[t] = pol.select_best(bels, t)
            r[t] = R[i[t], t]
            bels[i[t]] = bels[i[t]].update(r[t])
        choices.append(i)
        regrets.append(np.cumsum(r_best - r))
        discounted_rewards.append(np.sum(r * discount_seq))
    regrets, choices = np.array(regrets), np.array(choices)
    discounted_rewards = np.array(discounted_rewards)

    return EvalResult(regrets, choices, discounted_rewards)

class EvalResult(object):

    def __init__(self, regrets, choices, discounted_reward):
        self.regrets = regrets[:]
        self.choices = choices[:]
        self.discounted_reward=discounted_reward[:]
        self.regrets.flags.writeable=False
        self.choices.flags.writeable=False
        self.discounted_reward.flags.writeable=False

    def get(self):
        return (self.regrets, self.choices, self.discounted_reward)
        
    def __hash__(self):
        self.regrets.flags.writeable=False
        self.choices.flags.writeable=False
        return hash((self.regrets.data, self.choices.data, self.disounted_reward))

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, choices=['bernoulli', 'normal'])
    parser.add_argument('n_trials', type=int)
    parser.add_argument('T', type=int, default=20)
    parser.add_argument('K', type=int, default=5)
    parser.add_argument('--pgi_N', type=int, nargs='*', default=[100])
    parser.add_argument('--pgi_T', type=int, nargs='*', default=[50])
    parser.add_argument('--gi', action='store_true')
    parser.add_argument('--bayesucb', action='store_true')
    parser.add_argument('--ucb', action='store_true')
    parser.add_argument('--thompson', action='store_true')
    parser.add_argument('--pfhgi', action='store_true')
    parser.add_argument('--fhgi', action='store_true')
    parser.add_argument('--pbayesucb', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--parallel', type=int, default=0)
    parser.add_argument('--fixedmu', action='store_true')
    parser.add_argument('--reseed', action='store_true')
    parser.add_argument('--timing_analysis', action='store_true')
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_arguments()


    if args.type=='bernoulli':
        prior = Beta((1, 1))
    elif args.type=='normal':
        prior = NormalInverseGamma(0, 1, 1, 1)

    m = MAB(prior, args.K)

    if args.fixedmu:
        raise NotImplemented
        m = MAB(prior, 2, [Bernoulli(0.1), Bernoulli(0.2)])

    outf = h5py.File('.results.{}.h5'.format(args.type), 'a')
    run_k = str((args.n_trials, args.T, args.K, args.fixedmu))
    
    pol_cfg = {'pgi_N': args.pgi_N,
               'pgi_T': args.pgi_T,
               'pfhgi': args.pfhgi,
               'fhgi': args.fhgi,
               'gi_gamma': [0.9],
               'gi': args.gi,
               'bayesucb': args.bayesucb,
               'ucb': args.ucb,
               'thompson': args.thompson,
               'pbayesucb': args.pbayesucb,
               'H': args.T,
               'use_cache': not args.timing_analysis,
               'run_k': run_k,
               'run_type': args.type}

    if args.reseed or 'seed' not in outf:
        for g in outf:
            del outf[g]
        outf['seed'] = int(time.time())

    seed = outf['seed'][()]

    mean_g = outf.create_group('mean') if 'mean' not in outf else outf['mean']
    time_g = outf.create_group('time') if 'time' not in outf else outf['time']
    discount_g = outf.create_group('discounted_reward') if 'discounted_reward' \
        not in outf else outf['discounted_reward']

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
        import sys
        sys.exit(0)

    mean, variance, err_rate, discounted = mab_eval(m, args.T, pol_cfg, N_trials=args.n_trials, seed=seed, parallel=args.parallel)
    


    print "Regret"
    for i, k in enumerate(params_list):
        print "{} mean: ".format(k), mean[i]
        mean_g[k][run_k] = mean[i]

    print "Discounted"
    for i, k in enumerate(params_list):
        print "{} mean: ".format(k), discounted[i]
        discount_g[k][run_k] = discounted[i]

