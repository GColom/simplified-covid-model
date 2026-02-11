#!/usr/bin/env python
# coding: utf-8
'''author: Giulio Colombini'''

import numpy as np
from matplotlib import pyplot as plt
from pymittagleffler import mittag_leffler 
from scipy.special import gamma

T_U     = 5.5
sigma_U = 2.3

alpha   = T_U**2/sigma_U**2
beta    = T_U   /sigma_U**2

print("Run with "+chr(945)+f" = {alpha:.3f}, "+chr(946)+f" = {beta:.3f}.")

def a(t, alpha, beta):
    return mittag_leffler(t**alpha, 
                                                         alpha, 1)

def gamma_distrib(t, alpha, beta):
    return beta**alpha * t**(alpha - 1) * np.exp(-beta*t) / gamma(alpha)

ts   = np.linspace(-1, 10, 5000)
mlvs = a(ts, alpha, beta).real

plt.plot(ts, mlvs, zorder = 0, linewidth = 2, 
         linestyle = 'dashed', label = r'$a(t, \alpha, \beta)$')

gvs = np.zeros_like(ts)

for k in range(1, 9):
    gvs += gamma_distrib(ts, k * alpha, beta)
    plt.plot(ts, gvs, zorder = -1, label = fr'$a_{k}(t, \alpha, \beta)$')

plt.xlim(ts.min(), ts.max())
#plt.ylim(0., 1.1*mlvs.max())
plt.xlabel('$t$', fontsize = 14)
plt.legend(fontsize = 10, ncols = 2)
plt.show()
