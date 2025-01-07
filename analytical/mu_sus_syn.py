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

c0_vals = np.linspace(0, 1, 5)[1:]
s0_vals = np.linspace(0, 1., 1000)[1:]

mus = np.empty(shape = (*c0_vals.shape, *s0_vals.shape), dtype = float)
suss = np.empty(shape = (*c0_vals.shape, *s0_vals.shape), dtype = float)

for i, c0 in enumerate(c0_vals):
    for j, s0 in enumerate(s0_vals):
        mus[i,  j] = find_root(c0, s0).root
        suss[i, j] = sus(mus[i,j], beta, c0, s0, T_u, T_e, alpha, a)


fig, axes = plt.subplots(2,1, figsize = (6 + 2/3, 2 * (6+2/3)/np.sqrt(2)))

for i in range(len(c0_vals)):
    axes[0].plot(s0_vals, mus[i,:], label = rf"$n_0 = {c0_vals[i]}$")
    axes[1].plot(s0_vals, suss[i,:], label = rf"$n_0 = {c0_vals[i]}$")
axes[0].legend(fontsize = 12)
axes[1].legend(fontsize = 12)
axes[1].set_xlabel(r"$s$", fontsize = 12)
axes[0].set_ylabel(r"$\mu$", fontsize = 12)
axes[1].set_ylabel(r"$\mathrm{d} \mu /\mathrm{d} s$", fontsize = 12)
plt.tight_layout()
plt.show()
