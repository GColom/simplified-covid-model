#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:38:19 2022

@author: Giulio Colombini
"""

import numpy as np
from scipy.stats import binom

# SIMULATION PARAMETERS

t_end = 2500

# MODEL PARAMETERS

# Effective group population size
Ng    = 100
# Cooperator prize
kappa = 3.
# Defector prize
delta = 20 

# Defectors beta
betaD = 1./1.2
# Cooperator beta
betaC = 0.2*betaD

# "Convenience factor"
g = kappa/delta

def pstar(g):
    return max(0., min(1., 0.5/(1 - g)))

nc   = np.empty(shape = (t_end))

for t in range(t_end):
    pstar_ = pstar(g)
    Nc = binom.rvs(n=Ng, p = pstar_, size = 1)
    nc[t] = Nc/Ng

beta = betaC * nc + (1. - nc)*betaD

from matplotlib import pyplot as plt

figpstar, axpstar= plt.subplots()
plotrange = np.linspace(0,1, 1000)

plt.plot(plotrange, [pstar(g) for g in plotrange])
plt.scatter(g, pstar(g), marker = "*", color = "r", zorder = -1, linewidth = 0.5)
axpstar.set_xlabel(r"$g$", fontsize = 12)
axpstar.set_ylabel(r"$p^*$", fontsize = 12)

figbeta, axbeta = plt.subplots()

plt.plot(beta)

plt.hlines(y=betaC, xmin=0, xmax=t_end, linewidth=2, color='r', zorder = -1)
plt.hlines(y=betaD, xmin=0, xmax=t_end, linewidth=2, color='b', zorder = -1)
axbeta.set_ylabel(r"$\beta$", fontsize = 12)

axnc = axbeta.twinx()
axnc.plot(nc, color = "C1", linewidth = 0.5)
axnc.set_ylabel(r"$n_c$", fontsize = 12)
plt.show()
