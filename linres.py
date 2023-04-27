#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:52:51 2022

@author: Giulio Colombini
"""

import model as m
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

T_u     = 5.5
sigma_u = 2.3
T_e     = 2.0
sigma_e = 0.1
beta    = 1./1.2
m_base  = 0.4
c_0     = 1.0 
alpha   = .14

a       = T_u**2/sigma_u**2
b       = T_u/sigma_u**2

def sceq_mu(x, beta, c_0, m, T_u, T_e, alpha, a):
    return (1-(a**a)/(x+a)**a)*np.exp(-T_e*x/T_u) - x/(T_u*(1-alpha)*c_0*beta*m)

def Fprime(x, beta, c_0, m, T_u, T_e, alpha, a):
    return np.exp(-T_e/T_u * x) * ((a/(x+a))**a * ( a/(x+a) + T_e/T_u) - T_e/T_u)

def sus(x, beta, c_0, m, T_u, T_e, alpha, a):
    #print(x/(m - T_u * beta * (1 - alpha) * c_0 * m**2 * Fprime(x, beta, c_0, m, 
    #                                                               T_u, T_e, alpha, a)))
    return x/(m - T_u * beta * (1 - alpha) * c_0 * m**2 * Fprime(x, beta, c_0, m, 
                                                                   T_u, T_e, alpha, a))
def sus_num(x, beta, c_0, m, T_u, T_e, alpha, a):
    return x

def sus_denom(x, beta, c_0, m, T_u, T_e, alpha, a):
    return (m - T_u * beta * (1 - alpha) * c_0 * m**2 * Fprime(x, beta, c_0, m, 
                                                                   T_u, T_e, alpha, a))

# plt.figure()
# plt.plot(np.linspace(-1.5,2,1000), sceq_mu(np.linspace(-1.5,2,1000), 
#                                            beta, c_0, m_base, T_u, T_e, alpha, a))
# plt.grid(True)
# plt.show()
def find_root(c_0):
    try:
        ret = root_scalar(sceq_mu, bracket = (- 10, -0.000001, ), method = 'bisect',
                    args=(beta, c_0, m_base, T_u, T_e, alpha, a))
    except ValueError:
        ret = root_scalar(sceq_mu, bracket = (0.000001, 10., ), method = 'bisect',
                    args=(beta, c_0, m_base, T_u, T_e, alpha, a))
    return ret

dt = 1./48

offset   = 110
duration = 20


m_0   = (np.array([0, np.inf]), np.array([m_base, m_base]))
gamma = (np.array([0,np.inf]), np.array([0.,0.]))
 
sim_0 = m.run_simulation(days = 350, m = m_0, N = 1e6, dt = dt, norm = True, gamma = gamma)

pert_0 = m.simulate_perturbation(offset, duration, m_0, 0.95, N = 1e6, norm = True, dt = dt,
                                 gamma = gamma)
pert_1 = m.simulate_perturbation(offset, duration, m_0, 1.05, N = 1e6, norm = True, dt = dt,
                                 gamma = gamma)

fig1, ax1 = plt.subplots()
 
ax1.plot(dt * sim_0[0], sim_0[2], label = '$E_0$', color = "C0")
ax1.plot(dt * sim_0[0], sim_0[3], label = '$I_0$', color = "C1")
# ax1.plot(dt * sim_0[0], sim_0[4], label = '$H_0$', color = "C2")
ax1.plot(dt * sim_0[0], sim_0[5], label = '$A_0$', color = "C3")

deviation_mask_0 = np.logical_and(dt * pert_0[0] >= offset, dt * pert_0[0] <= offset + duration)
deviation_mask_1 = np.logical_and(dt * pert_1[0] >= offset, dt * pert_1[0] <= offset + duration)

ax1.plot(dt * pert_0[0][deviation_mask_0], pert_0[2][deviation_mask_0], label = '$E_{-5\%}$', 
        linewidth = 0, marker = '$-$', color = "C0")
ax1.plot(dt * pert_0[0][deviation_mask_0], pert_0[3][deviation_mask_0], label = '$I_{-5\%}$', 
        linewidth = 0, marker = '$-$', color = "C1")
#ax1.plot(dt * pert_0[0][deviation_mask_0], pert_0[4][deviation_mask_0], label = '$H_{95\%}$', 
#         linewidth = 0, marker = '$-$', color = "C2")
ax1.plot(dt * pert_0[0][deviation_mask_0], pert_0[5][deviation_mask_0], label = '$A_{-5\%}$', 
        linewidth = 0, marker = '$-$', color = "C3")

ax1.plot(dt * pert_1[0][deviation_mask_1], pert_1[2][deviation_mask_1], label = '$E_{+5\%}$', 
        linewidth = 0, marker = '$+$', color = "C0")
ax1.plot(dt * pert_1[0][deviation_mask_1], pert_1[3][deviation_mask_1], label = '$I_{+5\%}$', 
        linewidth = 0, marker = '$+$', color = "C1")
#ax1.plot(dt * pert_1[0][deviation_mask_1], pert_1[4][deviation_mask_1], label = '$H_{105\%}$', 
#        linewidth = 0, marker = '$+$', color = "C2")
ax1.plot(dt * pert_1[0][deviation_mask_1], pert_1[5][deviation_mask_1], label = '$A_{+5\%}$', 
        linewidth = 0, marker = '$+$', color = "C3")


ax1.set_ylim(bottom = 0.)
ax1.set_xlim(left = 0., right = dt * sim_0[0].max())


c = sim_0[1]/(sim_0[1]+sim_0[2]+sim_0[5]+sim_0[6])

with open("./c_dump.txt", mode= "w") as fdump:
    np.savetxt(fdump,c)

r_objs = np.array([find_root(element) for element in c])
mus    = np.array([ro.root for ro in r_objs])


susceptivities = np.zeros(shape = (len(mus)))
numerators     = np.zeros(shape = (len(mus)))
denominators   = np.zeros(shape = (len(mus)))
for i in range(len(mus)):
    susceptivities[i] = sus(mus[i], beta, c[i], m_base, T_u, T_e, alpha, a)
    numerators[i]     = sus_num(mus[i], beta, c[i], m_base, T_u, T_e, alpha, a)
    denominators[i]   = sus_denom(mus[i], beta, c[i], m_base, T_u, T_e, alpha, a)
    print("mu = ", mus[i]," dmu/dm = ", susceptivities[i])
    

ax2 = ax1.twinx()
ax2.plot(sim_0[0]*dt, mus , label = "$\mu$", linestyle = "dashed", color = "C2")
ax2.plot(sim_0[0]*dt, susceptivities, label = r"$\frac{d\mu}{dm}$", 
                      linestyle = "dashed", color = "C3")
ax2.plot(sim_0[0]*dt, c, label = "$c_0(t)$", linestyle = "dashed", color = "C4")
ax2.plot(sim_0[0]*dt, denominators, label = r"$m (1 - c_0 T_u (1-\alpha ) \beta m)$",
                      linestyle = "dashed", color = "C5")

ax1.legend(loc = "upper left")
ax1.set_ylabel("Population fraction")
ax2.legend(loc = "upper right")
ax2.set_ylabel("$\mu$")
ax1.grid(True)
ax2.grid(True)
#plt.figure()

plt.figure()
fmus = np.linspace(-0.5, 0.5, 500)
plt.plot(fmus, sus(fmus, beta, c_0, m_base, T_u, T_e, alpha, a))
plt.grid()
plt.show()
