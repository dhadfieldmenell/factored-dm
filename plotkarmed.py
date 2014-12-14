import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

fname = '.results.{}.h5'

parser = argparse.ArgumentParser()
parser.add_argument('run_type', type=str, choices=['normal', 'bernoulli'])
parser.add_argument('n_trials', type=int)
parser.add_argument('T', type=int)
parser.add_argument('K', type=int)
parser.add_argument('--allow_missing', action='store_false')
args = parser.parse_args()

run_k = str((args.n_trials, args.T, args.K, False))

comparison_keys = {'IH GI 0.9': {'linestyle': ':'}, 
                   'FHGI' : {'linestyle': ':'}, 
                   # 'UCB': {'linestyle': '--'}, 
                   'Bayes-UCB': {'linestyle': '--'}, 
                   # 'Thompson Sampling': {'linestyle': '-.'},
                   "Particle GI {},{}".format((100, 50), .9) : {'linestyle': '-'},
                   # "Particle GI {},{}".format((100, 10), .9) : {'linestyle': '-'},
                   "Particle GI {},{}".format((100, 1), .9) : {'linestyle': '-'},
                   "Particle GI {},{}".format((10, 50), .9) : {'linestyle': '-'},
                   # "Particle GI {},{}".format((10, 10), .9) : {'linestyle': '-'},
                   # "Particle GI {},{}".format((10, 1), .9) : {'linestyle': '-'},
                   "Particle GI {},{}".format((50, 50), .9) : {'linestyle': '-'}}
                   # "Particle GI {},{}".format((50, 10), .9) : {'linestyle': '-'}}
                   # "Particle GI {},{}".format((50, 1), .9) : {'linestyle': '-'},}
                   # "Particle FHGI {}".format((100, 50)) : {'linestyle': '-'},
                   # "Particle FHGI {}".format((100, 10)): {'linestyle': '-'}}

if args.run_type=='normal':
    del comparison_keys['IH GI 0.9']
    del comparison_keys['FHGI']

f = h5py.File(fname.format(args.run_type), 'r')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for k in comparison_keys:
    if args.allow_missing:
        try:
            means_k = f['mean'][k][run_k][:]
        except KeyError:
            continue
    else:
        means_k = f['mean'][k][run_k][:]
    ax.plot(range(means_k.shape[0]), means_k, label="{}".format(k), lw=2.0, **comparison_keys[k])
plt.legend(loc='best')
plt.title("Regret for {}-armed {} bandit".format(args.K, args.run_type))
plt.savefig("figs/{}-{}.pdf".format(args.run_type, run_k))
# plt.show(block=False)

