#!/usr/bin/env python

from __future__ import division

import gurobipy as grb
GRB = grb.GRB # constants for gurobi

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



import sys, os
import re
eps = 10**-8
MAX_ITER=1000

"""
classes to solve a 1.5-armed bandit 

SVMRetVal is probably the one to use
currently simplest + fastest
"""

class SVMRetVal(object):
    
    def __init__(self, gamma, feats, T, N, FH=np.inf, 
                 tol=1e-3):
        """
        FH indicate the length of the finite horizon. Defaults to None
        which is an infinite horizon criterion.
        """
        self.gamma = gamma
        self.feats = feats
        self.T     = T
        self.N     = N
        self.FH    = FH
        self.clf   = SVC(C=10)        
        self.tol   = tol

        self.mu    = np.zeros(N)
        self.phi   = np.zeros( (N, T, feats.get_size()) )
        # -1 indicates never retire
        self.tau   = np.ones(N) * -1 

    def set_particle(self, n, mu, R, update=False):
        self.mu[n] = mu
        for i in range(1, self.T):
            self.phi[n, i] = self.feats.feature(R[:i])
        
    def solve(self):
        max_nu = np.max(self.mu)
        min_nu = np.mean(self.mu)
        nu = (max_nu + min_nu) / 2
        while (max_nu - min_nu) > self.tol:            
            if self.ret_val(nu) > nu:
                # below GI
                min_nu = nu
            else: # self.ret_val(nu) == nu
                max_nu = nu
            nu = (max_nu + min_nu) / 2
        return nu

    def ret_val(self, lmbda):
        # it is always optimal to retire
        # if the rate is less than lmbda
        Y = self.mu > lmbda
        # the per-round cost of making an incorrect 
        # decision. It seems like this should be asymetric, but 
        # for some reason this performs better (in terms of MSE for Bernoulli)
        # Cost for retiring early is 
        # (self.mu - lmda) / (1-gamma)
        weights = np.abs(self.mu - lmbda)
        T = min(self.T, self.FH)
        # default to never stopping
        self.tau = np.ones(self.N, dtype=np.int) * T

        alive = np.ones(self.N, dtype=np.bool)
        denoms = np.cumsum(np.power(self.gamma, xrange(T)))
        for t in range(1, T):
            if len(np.unique(Y[alive])) <=1:
                # we can terminate early
                break
            self.clf.fit(self.phi[alive, t], Y[alive], weights[alive])
            # predict 1 means continue
            retired = 1 - self.clf.predict(self.phi[alive, t])
            # keep track of when particles retire
            self.tau[alive] = retired*t
            alive = (self.tau == T)
        avg_reward = np.mean(self.mu * denoms[self.tau - 1])
        avg_time   = np.mean(denoms[self.tau - 1])

        return np.maximum(avg_reward / avg_rate, lmbda)                

def gi_outer_loop(ret_val_fn, tol=eps, max_iter = 500):
    nu = 0
    converged = False
    for i in xrange(max_iter):
        old_nu = nu
        nu = ret_val_fn(nu)
        if np.abs(old_nu - nu) < 1e-3:
            converged = True
            break
    return nu

def fh_bernoulli_ret_val(rho, H, (a, b), tol=eps):
    l = rho / H
    a_vals = np.linspace(0, H, H + 1) + a    
    Vt = np.zeros(H+1)
    if H == 0:
        return rho
    # Take the horizion from H-1 to 0    
    for cur_h in range(H, 0, -1):
        # P[i] = i / Nt; i in [0,...,cur_h]
        cur_confidence = cur_h + a + b - 1
        P = a_vals[:cur_h] / (cur_confidence)

        # do a value iteration backup
        # Vt_minus1[i] = P(tails)*0 + P(heads)*1 
        #         + (P(tails) * Vt[i] + P(heads) * Vt[i+1])
        Vt = np.maximum(P + (Vt[:-1]*(1-P) + Vt[1:]*(P)), l * (H - cur_h))
    return Vt[0]    

def bernoulli_ret_val(rho, gamma, (a, b), tol=eps):
    # Set horizon to get tolerence right
    log_V_ub = np.log(1) - np.log(1 - gamma)
    H_lb = ( np.log(tol) - log_V_ub ) / np.log(gamma)
    H = int(np.ceil(H_lb))    
    a_vals = np.linspace(0, H, H + 1) + a    
    Vt = np.zeros(H+1)

    # Take the horizion from H-1 to 0    
    for cur_h in range(H, 0, -1):
        # P[i] = i / Nt; i in [0,...,cur_h]
        cur_confidence = cur_h + a + b - 1
        P = a_vals[:cur_h] / (cur_confidence)

        # do a value iteration backup
        # Vt_minus1[i] = P(tails)*0 + P(heads)*1 
        #         + gamma * (P(tails) * Vt[i] + P(heads) * Vt[i+1])
        Vt = np.maximum(P + gamma * (Vt[:-1]*(1-P) + Vt[1:]*(P)), rho)

    return Vt[0]

################################################################################
## Old approaches
################################################################################

class ParticleRetALP(object):

    """
    computes an estimate of a retirement process value function

      min          rho   + sum(V) + C * w.dot(w) + D * sum(abs(xi))
 w, V, b, rho, xi

      s.t.        V[t, n] >= rho
                  V[t, n] >= w[t, :].dot(phi[t, n]) + b[t]
                  w.dot(phi[t, n]) + b[t] = mu[n] + gamma * V[t+1, n] + pos_xi[t, n] - neg_xi[t, n]

                  rho >= sum( mu + gamma * V[1, :] ) / N

    This can be thought of as doing a time-varying linear approximation 
    to the value function

    Note the reference to mu! We are sampling across time steps to
    estimate the distribution of belief states. We are using
    the true value of the (sampled) latent parameters to specify the 
    Bellman constraints

    This is pretty slow and not really recommended. Kept around because
    it may prove useful for certain classes of problems
    """
    
    def __init__(self, gamma, feats, T, N,
                 C=1, D=100, verbose=False):
        self.model = grb.Model()        
        self.model.setParam('BarConvTol', 1e-2)
        self.init_solve=True
    
        self.T = T
        self.gamma = gamma
        self.feats = feats
        self.C = C
        self.D = D
        self.N = N
        self.mu = np.zeros(N)

        self.rho = self.model.addVar(lb = -1*GRB.INFINITY, name='rho', obj=1)

        self.V = np.array([[self.model.addVar(lb=-1*GRB.INFINITY, name='V_{}'.format((t, n)), obj=1)
                            for n in range(N)]
                           for t in range(1, T)])
                  
        self.w = np.array([[self.model.addVar(lb=-1*GRB.INFINITY, name='w_{}'.format((t, i)), obj=0)
                            for i in range(feats.get_size())]
                           for t in range(1, T)])
                           
        
        self.b = np.array([self.model.addVar(lb=-1*GRB.INFINITY, name='b_{}'.format(t), obj=0) 
                           for t in range(1, T)])
        
        self.pos_xi = np.array([[self.model.addVar(lb=0, name='xi_pos_{}'.format((t, n)), obj=self.D) 
                                 for n in range(N)]
                                for t in range(1, T)])
        self.neg_xi = np.array([[self.model.addVar(lb=0, name='xi_neg_{}'.format((t, n)), obj=self.D) 
                                 for n in range(N)]
                                for t in range(1, T)])

        self.model.update()

        obj = self.C * grb.quicksum([np.dot(self.w[i], self.w[i]) for i in range(len(self.w))])
        obj += self.model.getObjective()
        self.model.setObjective(obj)

        # this will get overwritten
        # we just initialize with something so that the value of the optimization is always finite
        self.rho_constr = self.model.addConstr(self.rho >= 0, name='rho_lb')

        #  equal to the current estiamte of the value for each particle
        for t in range(self.V.shape[0]):
            for n in range(self.V.shape[1]):
                if self.gamma < 1:
                    # use infinite horizon objective
                    self.model.addConstr(self.V[t, n] >= self.rho)
                else:
                    # use finite horizon
                    # rho is replaced with rho/T
                    self.model.addConstr(self.V[t, n] >= self.rho * (T-t-1) / T )
        self.model.update()
        # so we can update them
        self.particle_constrs = dict([(n, []) for n in range(self.N)])

    def set_particle(self, n, mu_n, R_n, update=False):
        """
        mu: mean of the sampled latent parameters
        R: sample of rewards from 0:T-1
        """
        self.mu[n] = mu_n
        for c in self.particle_constrs[n]:
            self.model.remove(c)
        self.particle_constrs[n] = []

        for t in range(1, self.T):
            idx = t-1 # because we don't have variables for t=0
            phi_t = self.feats.feature(R_n[:t])
            Q_tn_coeffs = [(p, w) for p, w in zip(phi_t, self.w[idx])]
            Q_tn_coeffs.append((1, self.b[idx]))
            Q_tn = grb.LinExpr(Q_tn_coeffs)
            if t < self.T-1:
                rhs_coeffs = [(self.gamma, self.V[t, n])]
            else:
                rhs_coeffs = []
            rhs_coeffs.extend([(1, self.pos_xi[idx, n]),
                               (-1, self.neg_xi[idx, n])])
            rhs = grb.LinExpr(rhs_coeffs)
            rhs += mu_n
            # w'phi_t == mu[n] + gamma*V(t+1)
            self.particle_constrs[n].append(
                self.model.addConstr(Q_tn == rhs))
            # V(t) >= w'phi_t
            self.particle_constrs[n].append(
                self.model.addConstr(self.V[idx, n] >= Q_tn))

        if update:
            self.model.remove(self.rho_constr)

            # 1/N sum( gamma * V[1, n] )
            if self.T > 1:
                rhs_coeffs = [(self.gamma / self.N, V_n) for V_n in self.V[0]]
                rhs = grb.LinExpr(rhs_coeffs)
            else:
                rhs = grb.LinExpr()
            # inequalities and it is definitely necessary here
            rhs += np.mean(self.mu)
            self.rho_constr = self.model.addConstr(self.rho >= rhs, name='rho_lb')
            self.model.update()

    def solve(self, verbose=False):
        self.model.setParam('OutputFlag', verbose)
        self.model.update()
        self.model.optimize()
        if self.gamma == 1:
            return self.rho.X / (self.T)
        else:
            return self.rho.X * (1-gamma)

class ParticleRetALPv2(ParticleRetALP):

    """
    Same as above but with only V's in the value iteration constraint
    and the linear Q-function enforcing that V's within a timestep
    are close to a linear combination of features

      min          rho   + sum(V) + C * w.dot(w) + D * sum(abs(xi))
 w, V, b, rho, xi

      s.t.        V[t, n] >= rho
                  V[t, n] >= mu[n] + gamma * V[t+1, n]
                  w.dot(phi[t, n]) + b[t] = mu[n] + gamma * V[t+1, n] + pos_xi[t, n] - neg_xi[t, n]

                  rho >= sum( mu + gamma * V[1, :] ) / N
    """
    
    def __init__(self, gamma, feats, T, N,
                 C=1, D=100, verbose=False):
        self.model = grb.Model()   
        self.model.setParam('OutputFlag', False)
        self.model.setParam('BarConvTol', 1e-2)
        self.init_solve=True
    
        self.T = T
        self.gamma = gamma
        self.feats = feats
        self.C = C
        self.D = D
        self.N = N
        self.mu = np.zeros(N)

        self.rho = self.model.addVar(lb = -1*GRB.INFINITY, name='rho', obj=1)

        self.V = np.array([[self.model.addVar(lb=-1*GRB.INFINITY, name='V_{}'.format((t, n)), obj=1)
                            for n in range(N)]
                           for t in range(1, T)])
                  
        self.w = np.array([[self.model.addVar(lb=-1*GRB.INFINITY, name='w_{}'.format((t, i)), obj=0)
                            for i in range(feats.get_size())]
                           for t in range(1, T)])
                           
        
        self.b = np.array([self.model.addVar(lb=-1*GRB.INFINITY, name='b_{}'.format(t), obj=0) 
                           for t in range(1, T)])
        
        self.pos_xi = np.array([[self.model.addVar(lb=0, name='xi_pos_{}'.format((t, n)), obj=self.D) 
                                 for n in range(N)]
                                for t in range(1, T)])
        self.neg_xi = np.array([[self.model.addVar(lb=0, name='xi_neg_{}'.format((t, n)), obj=self.D) 
                                 for n in range(N)]
                                for t in range(1, T)])

        self.model.update()

        obj = self.C * grb.quicksum([np.dot(self.w[i], self.w[i]) for i in range(len(self.w))])
        obj += self.model.getObjective()
        self.model.setObjective(obj)

        # this will get overwritten
        # we just initialize with something so that the value of the optimization is always finite
        self.rho_constr = self.model.addConstr(self.rho >= 0, name='rho_lb')

        #  equal to the current estiamte of the value for each particle
        for t in range(self.V.shape[0]):
            for n in range(self.V.shape[1]):
                if self.gamma < 1:
                    # use infinite horizon objective
                    self.model.addConstr(self.V[t, n] >= self.rho)
                else:
                    # use finite horizon
                    # rho is replaced with rho/T
                    self.model.addConstr(self.V[t, n] >= self.rho * ((T-t-1) / T) )
        self.model.update()
        # so we can update them
        self.particle_constrs = dict([(n, []) for n in range(self.N)])

    def set_particle(self, n, mu_n, R_n, update=False):
        """
        mu: mean of the sampled latent parameters
        R: sample of rewards from 0:T-1
        """
        self.mu[n] = mu_n
        for c in self.particle_constrs[n]:
            self.model.remove(c)
        self.particle_constrs[n] = []

        for t in range(1, self.T):
            idx = t-1 # because we don't have variables for t=0
            phi_t = self.feats.feature(R_n[:t])
            Q_tn_coeffs = [(p, w) for p, w in zip(phi_t, self.w[idx])]
            Q_tn_coeffs.append((1, self.b[idx]))
            Q_tn = grb.LinExpr(Q_tn_coeffs)
            if t < self.T-1:
                rhs_coeffs = [(self.gamma, self.V[t, n])]
            else:
                rhs_coeffs = []
            rhs_coeffs.extend([(1, self.pos_xi[idx, n]),
                               (-1, self.neg_xi[idx, n])])
            rhs = grb.LinExpr(rhs_coeffs)
            rhs += mu_n
            # w'phi_t == mu[n] + gamma*V(t+1)
            self.particle_constrs[n].append(
                self.model.addConstr(Q_tn == rhs))
            # V(t) >= mu + gamma * V(t+1)
            self.particle_constrs[n].append(
                self.model.addConstr(self.V[idx, n] >= rhs))

        if update:
            self.model.remove(self.rho_constr)

            # 1/N sum( gamma * V[1, n] )
            if self.T > 1:
                rhs_coeffs = [(self.gamma / self.N, V_n) for V_n in self.V[0]]
                rhs = grb.LinExpr(rhs_coeffs)
            else:
                rhs = grb.LinExpr()
            # inequalities and it is definitely necessary here
            rhs += np.mean(self.mu)
            self.rho_constr = self.model.addConstr(self.rho >= rhs, name='rho_lb')
            self.model.update()
