#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:52:51 2022

@author: Giulio Colombini
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import root_scalar

T_u     = 5.5
sigma_u = 2.3
T_e     = 2.0
sigma_e = 0.1
beta    = 1./1.2
m_base  = 0.7
c_0     = 1.0 
alpha   = .14

a       = T_u**2/sigma_u**2
b       = T_u/sigma_u**2

def sceq_mu(x, beta, c_0, m, T_u, T_e, alpha, a):
    return (1-(a**a)/(x+a)**a)*np.exp(-T_e*x/T_u) - x/(T_u*c_0*beta*m)

def Fprime(x, beta, c_0, m, T_u, T_e, alpha, a):
    return np.exp(-T_e/T_u * x) * ((a/(x+a))**a * ( a/(x+a) + T_e/T_u) - T_e/T_u)

def sus(x, beta, c_0, m, T_u, T_e, alpha, a):
    #print(x/(m - T_u * beta * (1 - alpha) * c_0 * m**2 * Fprime(x, beta, c_0, m, 
    #                                                               T_u, T_e, alpha, a)))
    return x/(m - T_u * beta * c_0 * m**2 * Fprime(x, beta, c_0, m, T_u, T_e, alpha, a))
def sus_num(x, beta, c_0, m, T_u, T_e, alpha, a):
    return x

def sus_denom(x, beta, c_0, m, T_u, T_e, alpha, a):
    return (m - T_u * beta  * c_0 * m**2 * Fprime(x, beta, c_0, m, T_u, T_e, alpha, a))

def find_root(c_0, s):
    try:
        ret = root_scalar(sceq_mu, bracket = (- 10, -0.000001, ), method = 'bisect',
                    args=(beta, c_0, s, T_u, T_e, alpha, a))
    except ValueError:
        ret = root_scalar(sceq_mu, bracket = (0.000001, 10., ), method = 'bisect',
                    args=(beta, c_0, s, T_u, T_e, alpha, a))
    return ret

c0s = np.linspace(0, 1, 500)
ss  = np.linspace(0, 1.25, 500)

c0grid, sgrid = np.meshgrid(c0s, ss)

mus = np.empty(shape = (*c0s.shape,*ss.shape), dtype = float)
gridmus = np.empty(shape = c0grid.shape)

for i, c0 in enumerate(c0s):
    for j, s in enumerate(ss):
        mus[i, j] = find_root(c0, s).root
print(c0grid)
for i in range(c0grid.shape[0]):
    for j in range(c0grid.shape[1]):
        gridmus[i,j] = find_root(c0grid[i,j], sgrid[i,j]).root

fig, ax = plt.subplots(figsize = (6 + 2/3, (6+2/3)/np.sqrt(2)))
#plt.imshow(mus, cmap = "RdBu")
ax.set_xlim(left = 0.)
ax.set_ylim(bottom = 0., top = ss.max())
from matplotlib.colors import TwoSlopeNorm
print(gridmus.min(), gridmus.max())

import matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

shrunk_cmap = shiftedColorMap(matplotlib.cm.RdBu_r, start=0., 
                              midpoint=0.5, 
                              stop=0.5 + 0.5 * gridmus.max()/abs(gridmus.min()), name='shrunk')
m = ax.pcolormesh(c0grid, sgrid, gridmus, cmap = shrunk_cmap)
c = ax.contour(c0grid, sgrid, gridmus, colors = "black", levels = [0.] ,linestyles = "solid", negative_linestyles = "solid")#, norm = norm)
ax.clabel(c, fmt = {c.levels[0] : r"$\mu = 0$"}, inline=True, fontsize=10)

cbar = plt.colorbar(m) 
cbar.ax.set_ylabel(r"$\mu$", fontsize = 12)
ax.set_xlabel(r"$n_0$", fontsize = 12)
ax.set_ylabel(r"$s_0$", fontsize = 12)
plt.tight_layout()
plt.savefig("mu_heatmap.pdf", dpi = 300, transparent = True)
plt.savefig("mu_heatmap.png", dpi = 300, transparent = True)
plt.show()
