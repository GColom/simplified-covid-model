#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 10:38:19 2026

@author: Giulio Colombini
"""

import model as mdl
import matplotlib.pyplot as plt
import numpy as np
from pymittagleffler import mittag_leffler

from utilities import *

T_U = 5.5
sigma_U = 2.3

a = T_U**2/sigma_U**2
b = T_U/sigma_U**2

def k(t, alpha, beta):
    return beta**alpha * t**(alpha - 1) * mittag_leffler(beta**alpha * t**alpha, 
                                                         alpha, alpha)*np.exp(-beta*t)

dt   = 1./24.
norm = False

m = mdl.model(dt = dt, 
              beta = 1./1.2, 
              alpha = .14, 
              N = 886891, 
              norm = norm) 

s_test = [(np.array([0, 35, 60, np.inf]), np.array([1., 1., .1, .1])),
          (np.array([0, 35, 60, np.inf]), np.array([1., .25, .15, .5]))]


# Variant arrival schedule 
var_test = (np.array([  0, np.inf]), 
            np.array([1.0,  1.0]))

# Vaccination schedule in individuals/day. 
vacc_test = (np.array([  0, np.inf]), 
             np.array([ 0.,     0.]))

fig, ax = plt.subplots(2,1, figsize = (12,8), sharex = True)
plt.rcParams["figure.autolayout"] = True

for s_ in s_test:
    res = m.run(200, s = s_, variants = var_test, vaccines = vacc_test)
    t,s,e,u,i,r,tot,phi_se,phi_eu,phi_ui,phi_ur,phi_ir,phi_rs,phi_v = res

    #Test flow reconstruction
    
    kernel = k(t, a, b).real
    
    du = u[1:] - u[:-1]
    
    conv = np.convolve(kernel, du[:t.shape[0]])[:t.shape[0]] * 0.14 * dt
    
    #Graphics
    
    ax[0].plot(t, s, label = 'S', color = 'C0')
    ax[0].plot(t, e, label = 'E', color = 'C1')
    ax[0].plot(t, i, label = 'I', color = 'C2')
    ax[0].plot(t, u, label = 'U', color = 'C3')
    ax[0].plot(t, r, label = 'R', color = 'C4')
    
    newp = phi_ui[:t.size]
    
    ax[1].plot(t, newp, label = 'New positive cases')    
    ax[1].plot(t, conv, label = r'$a \ast \dot{U}(t)$ np.conv', color = 'red',
               linestyle = "dashed") 
    ax[1].set_xlabel('Days since the beginning of the epidemic', fontsize = 14)
    ax[1].set_ylabel('Individuals', fontsize = 14)
    #ax[1].set_ylim(bottom = 0)
    
    #ax[0].legend(fontsize = 14)
    #ax[1].legend(fontsize = 14, loc = "upper left")

    #if norm:
    #    ax[0].set_ylim(bottom = 0, top = 0.0025)
    #    ax[0].set_ylabel('Population Fraction', fontsize = 14)
    #else:
    #    ax[0].set_ylim(bottom = 0)#, top = 250000)
    #    ax[0].set_ylabel('Individuals', fontsize = 14)
    
    ax[0].set_xlim([0, max(t)])
    
    ax[0].vlines(var_test[0], *ax[0].get_ylim(), linestyle = 'dashed',
                 color = 'red', label = 'Variant arrival')
    ax[0].vlines(vacc_test[0], *ax[0].get_ylim(), linestyle = 'dashed',
                 color = 'blue', label = 'Beginning of vaccination')
plt.show()
    
