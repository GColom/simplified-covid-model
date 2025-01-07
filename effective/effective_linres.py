#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:52:51 2022

@author: Giulio Colombini
"""

import effective_model as m
import numpy as np
import matplotlib.pyplot as plt
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

def find_root(c_0):
    try:
        ret = root_scalar(sceq_mu, bracket = (- 10, -0.000001, ), method = 'bisect',
                    args=(beta, c_0, m_base, T_u, T_e, alpha, a))
    except ValueError:
        ret = root_scalar(sceq_mu, bracket = (0.000001, 10., ), method = 'bisect',
                    args=(beta, c_0, m_base, T_u, T_e, alpha, a))
    return ret

dt = 1./48

offset   = 45 
duration = 7 
end      = 90

startpoints = np.arange(offset, end, duration, dtype = int)

n_pert_runs = len(startpoints)

m_0   = (np.array([0, np.inf]), np.array([m_base, m_base]))
 
sim_0 = m.run_simulation(days = 110, m = m_0, N = 1e6, dt = dt, norm = True)

pertp = [] # Store +5% simulations
pertm = [] # Store -5% simulations

print()
print(f"Begin {n_pert_runs}, {duration} long sensitivity runs between {offset} and {end}.")
print()
print()

for i in range(n_pert_runs):
    print(f"Running {i+1} of {n_pert_runs}.")
    pertm.append(m.simulate_perturbation(startpoints[i], duration, m_0, 0.9, N = 1e6, 
                                         norm = True, dt = dt))
    pertp.append(m.simulate_perturbation(startpoints[i], duration, m_0, 1.1, N = 1e6, 
                                         norm = True, dt = dt))

print("Finished!")
fig1, ax1 = plt.subplots(figsize = (6 + 2/3, (6+2/3)/np.sqrt(2)))
 
reference_linewidth =1.5 

#ax1.plot(dt * sim_0[0], sim_0[2], label = '$E_0$', color = "C0", linewidth = reference_linewidth)
#ax1.plot(dt * sim_0[0], sim_0[3], label = '$I_0$', color = "C1")
#ax1.plot(dt * sim_0[0], sim_0[4], label = '$I_0$', color = "C2", linewidth = reference_linewidth)
ax1.plot(dt * sim_0[0], sim_0[4], label = '$U_0$', color = "C0", linewidth = reference_linewidth)

deviation_linewidth = 1.5 

for i in range(n_pert_runs):
    deviation_mask_p = np.logical_and(dt * pertm[i][0] >= startpoints[i], 
                                      dt * pertm[i][0] <= startpoints[i] + duration)
    deviation_mask_m = np.logical_and(dt * pertp[i][0] >= startpoints[i], 
                                      dt * pertp[i][0] <= startpoints[i] + duration)
    
    #ax1.plot(dt * pertm[i][0][deviation_mask_m], pertm[i][2][deviation_mask_m], 
    #         linewidth = deviation_linewidth, color = "C0")#, label = '$E_{-5\%}$')
    #ax1.plot(dt * pertm[i][0][deviation_mask_m], pertm[i][3][deviation_mask_m], 
              #  label = '$I_{-5\%}$',  linewidth = 0, marker = '$-$', color = "C1")
    #ax1.plot(dt * pertm[i][0][deviation_mask_m], pertm[i][4][deviation_mask_m], 
    #         linewidth = deviation_linewidth, color = "C2")#, label = '$I_{-5\%}$')
    ax1.plot(dt * pertm[i][0][deviation_mask_m], pertm[i][4][deviation_mask_m], 
             linewidth = deviation_linewidth, color = "C3")#, label = '$A_{-5\%}$')
    #ax1.plot(dt * pertp[i][0][deviation_mask_p], pertp[i][2][deviation_mask_p], 
    #         linewidth =  deviation_linewidth, color = "C0")#, label = '$E_{+5\%}$')
    #ax1.plot(dt * pertp[i][0][deviation_mask_p], pertp[i][3][deviation_mask_p], 
    #         label = '$I_{+5\%}$', linewidth = 0, marker = '$+$', color = "C1")
    #ax1.plot(dt * pertp[i][0][deviation_mask_p], pertp[i][4][deviation_mask_p], 
    #         linewidth = deviation_linewidth, color = "C2")#, label = '$I_{+5\%}$')
    ax1.plot(dt * pertp[i][0][deviation_mask_p], pertp[i][4][deviation_mask_p], 
             linewidth = deviation_linewidth, color = "C3")#, label = '$A_{+5\%}$')


ax1.set_ylim(bottom = 0., top =  0.4)
ax1.set_xlim(left = 30., right = 100)#dt * sim_0[0].max())


c = sim_0[1]/(sim_0[1]+sim_0[2]+sim_0[4]+sim_0[5])

# with open("./c_dump.txt", mode= "w") as fdump:
#     np.savetxt(fdump,c)

r_objs = np.array([find_root(element) for element in c])
mus    = np.array([ro.root for ro in r_objs])


susceptivities = np.zeros(shape = (len(mus)))
numerators     = np.zeros(shape = (len(mus)))
denominators   = np.zeros(shape = (len(mus)))
fprimes        = np.zeros(shape = (len(mus)))
for i in range(len(mus)):
    susceptivities[i] = sus(mus[i], beta, c[i], m_base, T_u, T_e, alpha, a)
    numerators[i]     = sus_num(mus[i], beta, c[i], m_base, T_u, T_e, alpha, a)
    denominators[i]   = sus_denom(mus[i], beta, c[i], m_base, T_u, T_e, alpha, a)
    fprimes[i]        = Fprime(mus[i], beta, c[i], m_base, T_u, T_e, alpha, a)
    #print("mu = ", mus[i]," dmu/dm = ", susceptivities[i])
    

ax2 = ax1.twinx()
ax2.plot(sim_0[0]*dt, mus , label = "$\mu$", linestyle = "dashed", 
         color = "C2", linewidth = 1.5)
ax2.plot(sim_0[0]*dt, susceptivities, label = r"$\frac{d\mu}{ds}$", 
                      linestyle = "-.", color = "C1")
ax2.plot(sim_0[0][:-2]*dt, (mus[1:] - mus[:-1])[1:] -(mus[1:] - mus[:-1])[:-1], label = "$\mu$", linestyle = "dashed", 
         color = "C2", linewidth = 1.5)
ax2.plot(sim_0[0]*dt, c, label = "$c_0(t)$", linestyle = "dashed", color = "C4")
ax2.plot(sim_0[0]*dt, denominators, label = r"$m (1 - c_0 T_u \beta m F'(\mu))$",
                      linestyle = "dashed", color = "C5")
ax2.plot(sim_0[0]*dt, fprimes, label = r"$F'(\mu)$",
                      linestyle = "dashed", color = "C6")

ax1.set_xlabel("$t$", fontsize = 12)
ax1.legend(loc = "upper left", fontsize = 12)
ax1.set_ylabel("Population fraction", fontsize = 12)
ax2.legend(loc = "upper right", fontsize = 12)
#ax2.set_ylabel(r"$\mu$", fontsize = 12, labelpad = 15)
ax2.set_ylabel(r"Dimensionless units", fontsize = 12, labelpad = 15)
#ax2.set_ylabel(r"$\dfrac{d\mu}{ds}$", fontsize = 12, rotation = 0, labelpad = 15)
ax2.set_ylim(bottom = -3, top = 3)
#ax2.set_ylim(bottom = 0.5, top = 2.)
#ax1.grid(True)
#ax2.grid(True)

#plt.figure()
#fmus = np.linspace(-0.5, 0.5, 500)
#plt.plot(fmus, sus(fmus, beta, c_0, m_base, T_u, T_e, alpha, a))
#plt.grid()
plt.tight_layout()
plt.savefig("sensitivity_run.pdf", dpi = 300, transparent = True)
plt.show()
