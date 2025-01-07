#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:38:19 2022

@author: Giulio Colombini
"""

import numpy as np
from   scipy import stats 
from   tqdm  import tqdm

# Global parameters to keep track of run parameters

_NORM_ = False
_DT_   = 1./48

# Social activity rate
# The example is a quite strong lockdown 30 days after the introduction of patient zero.
m_test = (np.array([0, 30, 60, np.inf]), np.array([1., .15, .15, .5]))

# Parameters of \rho_E over time
pars_e_test = (np.array([0., np.inf]), np.array([2.,2.]), np.array([.1,.1]))

# Parameters of \rho_I over time
pars_i_test = (np.array([0., np.inf]), np.array([21, 21]), np.array([2.3,2.3]))

# parameters of \rho_A over time
pars_a_test = (np.array([0., np.inf]), np.array([5.5,5.5]), np.array([2.3,2.3]))

def discrete_gamma(mean, std_dev, min_t = None, max_t = None):
    if (min_t == None) and (max_t == None):
        min_t = np.rint(mean) - np.rint(3 * std_dev)
        max_t = np.rint(mean) + np.rint(5 * std_dev)
    
    RV = stats.gamma(a = (mean/std_dev)**2, scale = (std_dev**2/mean))
    low_i, high_i = np.int32(np.round([min_t, max_t]))
    low_i = max([1, low_i])
    
    c = np.zeros(high_i, dtype = np.double)
    
    for j in range(low_i, high_i):
        c[j] = RV.cdf((j + 1/2)) - RV.cdf((j-1/2))
    c /= c.sum()
    return (c, low_i, high_i)

# Forward buffered convolution
def propagate_forward(t, max_t, donor, acceptors, kernel_tuple, branching_ratios = np.array([1.])):
    kernel, i0, lker = kernel_tuple
    if t + i0 > max_t:
        return
    if t + lker - 1 > max_t:
        k = kernel[i0 : max_t - t + 1]
        lk = len(k)
    else:
        k = kernel[i0:]
        lk = len(k)
    buffer = np.empty(shape = (lk,) + donor.shape)
    for i in range(lk):
        buffer[i] = donor * k[i]
    for a, r in zip(acceptors, branching_ratios):
        a[t + i0 : t + i0 + lk] += r * buffer

def run_simulation(days = 60, dt = _DT_, beta = 1/1.2, alpha = .14, 
                    N = 886891, norm = _NORM_, m = m_test,
                    pars_e = pars_e_test, pars_i = pars_i_test, 
                    pars_a = pars_a_test):
    '''
    Launch a simulation of the epidemic using the specified parameters.

    Parameters
    ----------
    days : float, optional
        Number of days to simulate. The default is 60.
    dt : float, optional
        Timestep, expressed as a fraction of day. The default is 1./24..
    beta : float, optional
        Infection probability. The default is 1/1.2.
    alpha : float, optional
        Probability of manifesting symptoms. The default is .14.
    N : float or int, optional
        Total population in the model. The default is 886891.
    norm : bool, optional
        Normalise populations if True, otherwise keep numbers unnormalised. The default is False.
    m : tuple of np.arrays, optional
        Days in which mobility is changed and the mobility value to consider 
        up to the next change. The last value in the days array MUST be np.inf, 
        with a repeated final value in the second one. The default is m_test.
    pars_e : tuple of np.arrays, optional
        Same rules as m but with format (days, means, standard_deviations) for the Exposed
        exit distribution. The default is pars_e_test.
    pars_i : tuple of np.arrays, optional
        Same as pars_e but for the Infected category. The default is pars_i_test.
    pars_a : tuple of np.arrays, optional
        Same as pars_e but for the Asymptomatic category. The default is pars_a_test.

    Returns
    -------
    t : np.array
        Simulation timestamps.
    S : np.array
        Time series for the Susceptibles compartment.
    E : np.array
        Time series for the Exposed compartment.
    I : np.array
        Time series for the Infected compartment.
    A : np.array
        Time series for the Asymptomatic compartment.
    R : np.array
        Time series for the Removed compartment.
    TOT : np.array
        Time series for the sum of all compartments, to check consistency.

    '''
    global _DT_, _NORM_
    _DT_ = dt
    _NORM_ = norm
    # Calculate number of iterations
    max_step = int(np.rint(days / dt))
    
    # Initialise compartments and flows memory locations
    S = np.zeros(max_step+1)
    E = np.zeros(max_step+1)
    I = np.zeros(max_step+1)
    A = np.zeros(max_step+1)
    R = np.zeros(max_step+1)
    TOT = np.zeros(max_step+1)
    
    Phi_SE = np.zeros(max_step+1)
    Phi_EA = np.zeros(max_step+1)
    Phi_AI = np.zeros(max_step+1)
    Phi_IR = np.zeros(max_step+1)
    Phi_AR = np.zeros(max_step+1)
    
    # Unpack parameter tuples and rescale them with dt.
    
    m_t    = m[0] / dt
    m_vals = m[1]
    
    m_array= np.array([m_vals[np.searchsorted(m_t, t, side = 'right') - 1] for t in range(max_step+1)])
    
    # Unpack distribution tuples and generate distributions
    
    rho_e_t      = pars_e[0] / dt
    rho_e_mus    = pars_e[1] / dt
    rho_e_sigmas = pars_e[2] / dt
    
    rho_es = [discrete_gamma(mu, sigma) for mu, sigma in zip(rho_e_mus, rho_e_sigmas)]

    rho_i_t      = pars_i[0] / dt
    rho_i_mus    = pars_i[1] / dt
    rho_i_sigmas = pars_i[2] / dt

    rho_is = [discrete_gamma(mu, sigma) for mu, sigma in zip(rho_i_mus, rho_i_sigmas)]

    rho_a_t      = pars_a[0] / dt
    rho_a_mus    = pars_a[1] / dt
    rho_a_sigmas = pars_a[2] / dt

    rho_as = [discrete_gamma(mu, sigma) for mu, sigma in zip(rho_a_mus, rho_a_sigmas)]
    
    # Add initial population to Susceptibles
    if norm:
        S[0]   = 1
        TOT[0] = 1
    else:
        S[0]   = N
        TOT[0] = N
    
    # Add patient zero to flow
    if norm:
        Phi_SE[0] += 1./N
    else:
        Phi_SE[0] += 1.
    
    # Intialize indices for distribution selection
    
    cur_rho_e_idx = 0
    cur_rho_i_idx = 0
    cur_rho_a_idx = 0
    
    # Master simulation loop
    for t in tqdm(range(max_step)):
        # Update distribution indices
        cur_rho_e_idx = np.searchsorted(rho_e_t, t, side = 'right') - 1
        cur_rho_i_idx = np.searchsorted(rho_i_t, t, side = 'right') - 1
        cur_rho_a_idx = np.searchsorted(rho_a_t, t, side = 'right') - 1

        # Get current parameters
        cur_m     = m_array[t]
        
        # Evaluate active population
        
        P = S[t] + E[t] + A[t] + R[t]
        
        # Evolve contagion flow

        Phi_SE[t] += beta * cur_m * S[t] * (A[t]) * dt / P
        
        # Propagate flows
        
        propagate_forward(t, max_step, Phi_SE[t], [Phi_EA], rho_es[cur_rho_e_idx],
                          branching_ratios = np.array([1.]))
        propagate_forward(t, max_step, Phi_EA[t], [Phi_AI, Phi_AR], rho_as[cur_rho_a_idx],
                          branching_ratios = np.array([alpha, 1. - alpha]))
        propagate_forward(t, max_step, Phi_AI[t], [Phi_IR], rho_is[cur_rho_i_idx],
                          branching_ratios = np.array([1.]))
        
        # Evolve compartments
        
        S[t+1] = S[t] - Phi_SE[t]
        E[t+1] = E[t] + Phi_SE[t] - Phi_EA[t]
        A[t+1] = A[t] + Phi_EA[t] - Phi_AI[t] - Phi_AR[t]
        I[t+1] = I[t] + Phi_AI[t] - Phi_IR[t]
        R[t+1] = R[t] + Phi_IR[t] + Phi_AR[t]
        TOT[t+1] = S[t+1] + E[t+1] + I[t+1] + A[t+1] + R[t+1]
    t = np.array([t for t in range(max_step+1)])
    return (t, S, E, I, A, R, TOT)

def simulate_perturbation(offset, duration, reference_m, m_perturbation_scaling, 
                          dt = _DT_, beta = 1/1.2, alpha = .14, 
                          N = 886891, norm = _NORM_, m = m_test, 
                          pars_e = pars_e_test, pars_i = pars_i_test, 
                          pars_a = pars_a_test):
    # First calculate total duration of simulation run
    total_duration = offset + duration
    
    # Then add a mobility breakpoint at the beginning of the perturbation
    reference_m_days, reference_m_vals = reference_m
    
    final_m_days = np.concatenate((reference_m_days[reference_m_days < offset], 
                                   np.array([offset]), 
                                   reference_m_days[reference_m_days > offset]))
    final_m_vals = np.concatenate((reference_m_vals[reference_m_days < offset],
                                   np.array([reference_m_vals[reference_m_days < offset][-1]]),
                                   reference_m_vals[reference_m_days > offset]))
    final_m_vals[final_m_days >= offset] *= m_perturbation_scaling

    print(final_m_days)
    print(final_m_vals)
    modified_m = (final_m_days, final_m_vals)
    return run_simulation(days = offset + duration, dt = dt, beta = beta, alpha = alpha, 
                          N = N, norm = norm, m = modified_m, 
                          pars_e = pars_e, pars_i = pars_i, 
                          pars_a = pars_a)

def test_model(days = 200, dt = 1/48, norm = False):
    print("Simulate", days, "days with a {:.2f}".format(dt), "day resolution.")
    print("The example is a quite strong lockdown 30 days after the introduction of patient zero.")
    t,s,e,i,a,r,tot = run_simulation(days = 200, dt = dt, norm = norm)
#%% Graphics
    from matplotlib import pyplot as plt
     
    plt.figure("Simulation test")
    plt.plot(t * dt,s, label = 'S', linewidth = .5)
    plt.plot(t * dt,e, label = 'E', linewidth = .5)
    plt.plot(t * dt,i, label = 'I', linewidth = .5)
    plt.plot(t * dt,a, label = 'A', linewidth = .5)
    plt.plot(t * dt,r, label = 'R', linewidth = .5)
    #plt.plot(t * dt, tot, label = 'TOT')
    
    plt.legend()
    plt.grid(True)
    if norm:
        plt.ylim(bottom = 0, top = 0.002255)
        plt.ylabel('Population Fraction')
    else:
        plt.ylim(bottom = 0, top = 2000)
        plt.ylabel('People')
    plt.xlim([0, max(t * dt)])
    plt.xlabel('Days since patient zero introduction')
    plt.ylabel('People')
    
    plt.show()

if __name__ == "__main__":
    test_model(norm = True)
