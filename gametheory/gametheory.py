#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:38:19 2022

@author: Giulio Colombini
"""

import numpy as np
from   scipy import stats 
from   scipy.optimize import root_scalar
from   tqdm  import tqdm

# SIMULATION PARAMETERS

_NORM_ = False 
_DT_   = 1./48

# MODEL PARAMETERS

# GAME PARAMETERS

# Effective group population size
Ng    = 100
# Cooperator prize
kappa = 3.
# Defector prize
delta = 2 

# Base beta 
beta0= 1./1.2
# Beta reduction factor for cooperators w.r.t. beta0.
R = 0.2

# Alarm threshold in percentage of population above which the government acts
alarm_threshold = 0.005

# Compute reduction factor w.r.t. defectors rate betaD.
red = R/(2-R)

# "Convenience factor"
g = kappa/delta

# EPIDEMIOLOGICAL PARAMETERS

T_u     = 5.5
sigma_u = 2.3
T_e     = 2.0
sigma_e = 0.1
T_i     = 21.0
sigma_i = 2.3

# Social activity rate
m_test = (np.array([0, np.inf]), np.array([1., 1.]))

# Parameters of \rho_E over time
pars_e_test = (np.array([0., np.inf]), np.array([T_e, T_e]), np.array([sigma_e, sigma_e]))

# Parameters of \rho_I over time
pars_i_test = (np.array([0., np.inf]), np.array([T_i, T_i]), np.array([sigma_i, sigma_i]))

# parameters of \rho_A over time
pars_a_test = (np.array([0., np.inf]), np.array([T_u, T_u]), np.array([sigma_u, sigma_u]))

# FUNCTIONS

def pstar(g):
    return max(0., min(1., 0.5/(1 - g)))

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

def run(days = 60, dt = _DT_, beta0 = beta0, red = red, alpha = .14, 
                  alarm_threshold = alarm_threshold,
                  g_open = 0.,
                  g_close= 0.6,
                  N = 886891, norm = _NORM_, m = m_test,
                  pars_e = pars_e_test, pars_i = pars_i_test, 
                  pars_a = pars_a_test, 
                  verbose = True, return_flows = False):
    '''
    Launch a simulation of the epidemic coupled to a game using the specified parameters.

    Parameters
    ----------
    days : float, optional
        Number of days to simulate. The default is 60.
    dt : float, optional
        Timestep, expressed as a fraction of day. The default is 1./24..
    betaC : float, optional
        Infection probability for collaborators. The default is 1/1.2.
    betaD : float, optional
        Infection probability for defectors. The default is 1/1.2.
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
    g0 : Initial ratio of collaborator/defector payoff. The default is 
    verbose : bool, optional
        Specifies whether a progress bar and the mobility schedule for the simulation
        should be printed upon calling the function. The default is True.
    return_flows: bool, optional
        Specifies whether the time series of the flows between compartments should be returned
        together with that of the compartments. The default is False.
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
    nc = np.zeros(int((max_step+1)*dt))
    
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
    
    # Init beta with beta0 at the beginning.
    beta = beta0

    def predict(t0, pred_length):
        print("IT'S PREDICTION DAY:", t0*dt)
        end_pred = min(max_step, t0+pred_length)
        # Update distribution indices
        cur_rho_e_idx = np.searchsorted(rho_e_t, t0, side = 'right') - 1
        cur_rho_i_idx = np.searchsorted(rho_i_t, t0, side = 'right') - 1
        cur_rho_a_idx = np.searchsorted(rho_a_t, t0, side = 'right') - 1

        # Get current parameters
        cur_m     = m_array[t0]

        #Function used by the government for weekly predictions.
        pred_S = S[:end_pred+1].copy()
        pred_E = E[:end_pred+1].copy()
        pred_I = I[:end_pred+1].copy()
        pred_A = A[:end_pred+1].copy()
        pred_R = R[:end_pred+1].copy()

        pred_Phi_SE = Phi_SE[:end_pred+1].copy()
        pred_Phi_EA = Phi_EA[:end_pred+1].copy()
        pred_Phi_AI = Phi_AI[:end_pred+1].copy()
        pred_Phi_IR = Phi_IR[:end_pred+1].copy()
        pred_Phi_AR = Phi_AR[:end_pred+1].copy()
         
        for t in range(t0, end_pred):
            # Evaluate active population 
            pred_P = pred_S[t] + pred_E[t] + pred_I[t] + pred_A[t] + pred_R[t]

            pred_Phi_SE[t] += beta * cur_m * pred_S[t] * (pred_A[t]) * dt / pred_P
            
            # Propagate flows
            
            propagate_forward(t, end_pred, pred_Phi_SE[t], [pred_Phi_EA], 
                              rho_es[cur_rho_e_idx],
                              branching_ratios = np.array([1.]))
            propagate_forward(t, end_pred, pred_Phi_EA[t], [pred_Phi_AI, pred_Phi_AR], 
                              rho_as[cur_rho_a_idx],
                              branching_ratios = np.array([alpha, 1. - alpha]))
            propagate_forward(t, end_pred, pred_Phi_AI[t], [pred_Phi_IR], 
                              rho_is[cur_rho_i_idx],
                              branching_ratios = np.array([1.]))
            
            # Evolve compartments
            
            pred_S[t+1] = pred_S[t] - pred_Phi_SE[t]
            pred_E[t+1] = pred_E[t] + pred_Phi_SE[t] - pred_Phi_EA[t]
            pred_A[t+1] = pred_A[t] + pred_Phi_EA[t] - pred_Phi_AI[t] - pred_Phi_AR[t]
            pred_I[t+1] = pred_I[t] + pred_Phi_AI[t] - pred_Phi_IR[t]
            pred_R[t+1] = pred_R[t] + pred_Phi_IR[t] + pred_Phi_AR[t]

        return (pred_S[t:t+pred_length], pred_E[t:t+pred_length], pred_A[t:t+pred_length],
                pred_I[t:t+pred_length], pred_R[t:t+pred_length])

    # Lockdown state variable. Start with open circulation 
    ld = False
    ld_history = np.zeros(max_step+1, dtype = bool)
    # Master simulation loop
    for t in tqdm(range(max_step), disable = not verbose):
        # SOCIABILITY GAME

        # Government turn
        # The government plays weekly.
        if np.fmod(t*dt, 7) == 0.:
            # First the government makes a prediction  
            I_nextweek = predict(t, int(7/dt))[-2][-1]
            thr = alarm_threshold if norm else alarm_threshold*N
            if I_nextweek >= thr:
                print("LOCKDOWN")
                g  = g_close 
                ld = True
            else:
                g  = g_open 
                ld = False
        ld_history[t] = ld 
        ## Individuals' turn
        # The individuals play once a day. 

        if np.fmod(t*dt, 1) == 0.:
            pstar_ = pstar(g)
            Nc = stats.binom.rvs(n=Ng, p = pstar_, size = 1)
            nc_ = Nc/Ng
            nc[int(t*dt)] = nc_ 
            beta = 2*beta0 * (1. + (red-1.)*nc_)/(1.+red)

        # EPIDEMIOLOGICAL MODEL

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
    return (t, S, E, I, A, R, TOT, nc, ld_history)
    
def test_model(days = 200, beta0 = 1/1.2, dt = 1/48, norm = True):
    print("Simulate", days, "days with a {:.2f}".format(dt), "day resolution.")
    t,s,e,i,a,r,tot,nc,lockdown_history = run(days = 200, beta0 = 1./1.2, dt = dt, norm = norm)
    
    # Find lockdown begin and end

    ld_chng = (np.logical_xor(lockdown_history[:-1], lockdown_history[1:]))
    ld_cpts = t[:-1][ld_chng]*dt
#%% Graphics
    from matplotlib import pyplot as plt
     
    plt.figure("Simulation test")
    #plt.plot(t * dt,s, label = 'S', linewidth = .5)
    plt.plot(t * dt,e, label = 'E', linewidth = .5)
    plt.plot(t * dt,a, label = 'A', linewidth = .5)
    plt.plot(t * dt,i, label = 'I', linewidth = .5)
    above_threshold = i[::int(7/dt)] > alarm_threshold
    plt.scatter(t[::int(7/dt)][above_threshold]*dt, 
                i[::int(7/dt)][above_threshold], 
                label = 'I', s=6, color ="red")
    plt.scatter(t[::int(7/dt)][np.logical_not(above_threshold)]*dt, 
                i[::int(7/dt)][np.logical_not(above_threshold)], 
                label = 'I', s=6, color = "green")
    #plt.plot(t * dt,r, label = 'R', linewidth = .5)
    for i, cpts in enumerate(zip(ld_cpts[:-1:2], ld_cpts[1::2])):
        if not i:
            plt.axvspan(cpts[0], cpts[1], alpha = 0.1, color = "red", label ="Lockdown periods")
        else:
            plt.axvspan(cpts[0], cpts[1], alpha = 0.1, color = "red")

    plt.hlines(y=alarm_threshold, xmin=0, xmax=days, linewidth=.5, color='r', zorder = -1,
               label = "Intervention threshold", linestyle = "dashed")
    #plt.plot(t * dt, tot, label = 'TOT')

    plt.xlim([0, max(t * dt)])
    plt.xlabel('Days since patient zero introduction', fontsize=12)
    plt.ylim(bottom = 0, top = a.max()*1.05)

    if norm:
        plt.ylabel('Population Fraction', fontsize=12)
    else:
        plt.ylabel('People', fontsize=12)
    
    #plt.legend()
    
    plt.tight_layout()
    plt.savefig("./pictures/gt.pdf", dpi = 300, transparent = True)

    figbeta, axbeta = plt.subplots()
    beta = 2*beta0 * (1. + (red-1.)*nc)/(1.+red)
    plt.plot(beta)
    
    plt.hlines(y=2*red*beta0/(1+red), xmin=0, xmax=days, linewidth=2, color='r', zorder = -1)
    plt.hlines(y=2 * beta0/(1+red), xmin=0, xmax=days, linewidth=2, color='b', zorder = -1)
    axbeta.set_ylabel(r"$\beta$", fontsize = 12)
    
    axnc = axbeta.twinx()
    axnc.plot(nc, color = "C1", linewidth = 0.5)
    axnc.set_ylabel(r"$n_c$", fontsize = 12)
    plt.show()

if __name__ == "__main__":
    test_model(norm = True)
