#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 4 10:38:19 2026

@author: Giulio Colombini
"""

import numpy as np
from scipy.optimize import root_scalar
from   tqdm  import tqdm
import pandas as pd
from pymittagleffler import mittag_leffler

from utilities import *

# Experimental data import

df = pd.read_csv('data/COVID_data.csv',
                index_col = 'datetime', 
                parse_dates = True, date_format = "%d/%m/%Y")

df.index = pd.to_datetime(df.index)
data_beg_date = df.index.min()
data_beg_day  = df.days.min()

print("Data begins on", data_beg_date, ', corresponding to day n.', data_beg_day)

new_positives = (df['new_positives'].rolling(7, center = False, min_periods = 1)
                                    .mean().to_numpy())

from matplotlib import pyplot as plt

plt.figure()
plt.plot(df.index, new_positives)
plt.show()

# Global parameters to keep track of run parameters

_NORM_ = False
_DT_   = 1./48

T_U     = 5.5
sigma_U = 2.3


# Transcendental equation that determines the Local Lyapunov exponent (LLE) mu.
def eq_mu(x, beta, n_0, m, T_u, T_e, alpha, a):
    return (1-(a**a)/(x+a)**a)*np.exp(-T_e*x/T_u) - x/(T_u*n_0*beta*m)

# Auxiliary function in the calculation of the LLE susceptivity.
def Fprime(x, beta, n_0, m, T_u, T_e, alpha, a):
    return np.exp(-T_e/T_u * x) * ((a/(x+a))**a * ( a/(x+a) + T_e/T_u) - T_e/T_u)

# Root finding for the LLE.
def mu(beta, n_0, s, T_u, T_e, alpha, a):
    #try:
    #    ret = root_scalar(eq_mu, bracket = (- 10, -1e-7, ), method = 'bisect',
    #                args=(beta, n_0, s, T_u, T_e, alpha, a))
    #except ValueError:
    #    ret = root_scalar(eq_mu, bracket = (1e-7, 10., ), method = 'bisect',
    #                args=(beta, n_0, s, T_u, T_e, alpha, a))
    ret = root_scalar(eq_mu, x0 = 0., x1 = 0.5, method = 'secant',
                    args=(beta, n_0, s, T_u, T_e, alpha, a))
    return ret.root

# Susceptivity computation.
def susceptivity(x, beta, n_0, s, T_u, T_e, alpha, a):
    return x/(s - T_u * beta * n_0 * s**2 * Fprime(x, beta, n_0, s, T_u, T_e, alpha, a))

def k(t, alpha, beta):
    return beta**alpha * t**(alpha - 1) * mittag_leffler(beta**alpha * t**alpha, 
                                                         alpha, alpha)*np.exp(-beta*t)
def lrt_Udot(t, t0, mu, dmuds, U0, T_U):
    kappa = U0*dmuds/T_U
    sigma = mu/T_U
    return kappa*np.exp(sigma*(t-t0))*(sigma*(t-t0) +1)

# Default parameter values

# Variant arrival schedule 
var_test = (np.array([  0, np.inf]), 
            np.array([1.0,  1.0]))

# Vaccination schedule in individuals/day. 
vacc_test = (np.array([  0, np.inf]), 
             np.array([ 0.,     0.]))

# Distribution \rho_E over time 
dist_e_test = (discrete_gamma,
               np.array([0., np.inf]), np.array([2.,2.]), np.array([.1,.1]))

# Distribution \rho_I over time 
dist_i_test = (discrete_gamma,
               np.array([0., np.inf]), np.array([21, 21]), np.array([2.3,2.3]))

# Distribution \rho_U over time
dist_u_test = (discrete_gamma,
               np.array([0., np.inf]), np.array([5.5,5.5]), np.array([2.3,2.3]))

# Distribution \rho_R over time
dist_r_test = (discrete_gamma,
               np.array([0., np.inf]), np.array([180, 180]), np.array([10,10]))

class model:
    def __init__(self, dt = 1./24., 
                 beta = 1./1.2, 
                 alpha = .14, 
                 N = 886891, 
                 norm = False, 
                 dist_e = dist_e_test, 
                 dist_i = dist_i_test, 
                 dist_u = dist_u_test, 
                 dist_r = dist_r_test):

        self.N      = N
        self.dt     = dt 
        print("init with dt =", self.dt)
        self.beta   = beta
        print("init with beta =", self.beta)
        self.alpha  = alpha
        self.norm   = norm
        self.dist_e = dist_e
        self.dist_i = dist_i
        self.dist_u = dist_u
        self.dist_r = dist_r
 
        self.max_step = 0 # Internal pointer to maximum time for which the contents
                          # of the flows are fully calculated, i.e. point from which
                          # the simulation must be resumed.

        # Placeholders for compartments and flows
        
        self.ts  = np.zeros(1, dtype = float)
        self.S   = np.zeros(1, dtype = float)
        self.E   = np.zeros(1, dtype = float)
        self.I   = np.zeros(1, dtype = float)
        self.U   = np.zeros(1, dtype = float)
        self.R   = np.zeros(1, dtype = float)
        self.TOT = np.zeros(1, dtype = float)

        self.Phi_SE = np.zeros(1, dtype = float)
        self.Phi_EU = np.zeros(1, dtype = float)
        self.Phi_UI = np.zeros(1, dtype = float)
        self.Phi_IR = np.zeros(1, dtype = float)
        self.Phi_UR = np.zeros(1, dtype = float)
        self.Phi_RS = np.zeros(1, dtype = float)
        self.Phi_V  = np.zeros(1, dtype = float) # Vaccination flow

        # Initialize first timestamp to 0.

        self.ts[0] = 0.

        # Add initial population to Susceptibles and patient 0 to flows.
        if self.norm:
            self.S[0]       = 1
            self.TOT[0]     = 1
            self.Phi_SE[0] += 1./N

        else:
            self.S[0]       = self.N
            self.TOT[0]     = self.N
            self.Phi_SE[0] += 1.

        # Unpack distribution tuples and generate distributions
        
        # Exposed
        self.rho_e        = dist_e[0]
        self.rho_e_t      = dist_e[1] / dt
        self.rho_e_mus    = dist_e[2] / dt
        self.rho_e_sigmas = dist_e[3] / dt
        
        self.rho_es = [self.rho_e(mu, sigma) for mu, sigma in zip(self.rho_e_mus, 
                                                             self.rho_e_sigmas)]

        # Isolated Infected 
        self.rho_i        = dist_i[0]
        self.rho_i_t      = dist_i[1] / dt
        self.rho_i_mus    = dist_i[2] / dt
        self.rho_i_sigmas = dist_i[3] / dt

        self.rho_is = [self.rho_i(mu, sigma) for mu, sigma in zip(self.rho_i_mus, 
                                                             self.rho_i_sigmas)]

        # Unreported
        self.rho_u        = dist_u[0]
        self.rho_u_t      = dist_u[1] / dt
        self.rho_u_mus    = dist_u[2] / dt
        self.rho_u_sigmas = dist_u[3] / dt

        self.rho_us = [self.rho_u(mu, sigma) for mu, sigma in zip(self.rho_u_mus, 
                                                             self.rho_u_sigmas)]
        
        # Removed 
        self.rho_r        = dist_r[0]
        self.rho_r_t      = dist_r[1] / dt
        self.rho_r_mus    = dist_r[2] / dt
        self.rho_r_sigmas = dist_r[3] / dt

        self.rho_rs = [self.rho_r(mu, sigma) for mu, sigma in zip(self.rho_r_mus, 
                                                             self.rho_r_sigmas)]

        # Intialize indices for distribution selection
        self.cur_rho_e_idx = 0
        self.cur_rho_i_idx = 0
        self.cur_rho_u_idx = 0
        self.cur_rho_r_idx = 0

    def run(self, t_end, s, variants, vaccines):

        # Run model from self.max_t to t_end using the sociability function s.
        # Return the compartments and flows.
        
        # First check that t_end is larger than the current time of the system.

        if t_end <= self.max_step * self.dt:
            raise ValueError("t_end must be larger than the current self.max_step*dt.")
        
        end_step = int(np.round(t_end / self.dt))

        # Initialize sociability and transimissivity arrays
        # from piecewise functions.
        # Unpack parameter tuples and rescale them with dt.

        s_t    = s[0] / self.dt
        s_vals = s[1]

        s_array= np.array([s_vals[np.searchsorted(s_t, t, side = 'right') - 1] 
                           for t in range(self.max_step, end_step+1)])
        
        tau_t    = variants[0] / self.dt
        tau_vals = variants[1]
        
        tau_array= np.array([tau_vals[np.searchsorted(tau_t, t, side = 'right') - 1] 
                             for t in range(self.max_step, end_step+1)])

        # Initialize vaccination rate time series.
        # Unpack parameter tuples and rescale them with dt.

        v_t     = vaccines[0] / self.dt
        v_vals  = vaccines[1]
       
        v_array = np.array([v_vals[np.searchsorted(v_t, t, side = 'right') - 1] 
                           for t in range(self.max_step, end_step+1)])
        
        # Compute the size of the arrays that will store the trajectory
        # and flows. For the flows we must pad the array with a tail 
        # that will store the partially evaluated convolution, in order
        # to allow the continuation of the simulation at a later time.
        
        compartments_size = max(0, end_step - self.max_step) + 1

        max_kernel_length = max(max(rho_tuple[2] for rho_tuple in self.rho_es),
                                max(rho_tuple[2] for rho_tuple in self.rho_us),
                                max(rho_tuple[2] for rho_tuple in self.rho_is),
                                max(rho_tuple[2] for rho_tuple in self.rho_rs))

        flows_size = compartments_size + max_kernel_length
        
        # Copy trajectory and flows to simulate system and simultaneously pad
        # memory locations to the desired.
        
        ts= np.array([idx for idx in range(self.max_step, end_step + 1)]) * self.dt 

        s = np.pad(self.S[self.max_step:end_step], 
                   (0, max(0, compartments_size-self.S[self.max_step:end_step].size)),
                   'constant', constant_values = 0.)
        e = np.pad(self.E[self.max_step:end_step], 
                   (0, max(0, compartments_size-self.E[self.max_step:end_step].size)),
                   'constant', constant_values = 0.)
        u = np.pad(self.U[self.max_step:end_step], 
                   (0, max(0, compartments_size-self.U[self.max_step:end_step].size)),
                   'constant', constant_values = 0.)
        i = np.pad(self.I[self.max_step:end_step], 
                   (0, max(0, compartments_size-self.I[self.max_step:end_step].size)),
                   'constant', constant_values = 0.)
        r = np.pad(self.R[self.max_step:end_step], 
                   (0, max(0, compartments_size-self.R[self.max_step:end_step].size)),
                   'constant', constant_values = 0.)
        tot = np.pad(self.TOT[self.max_step:end_step], 
            (0, max(0, compartments_size-self.TOT[self.max_step:end_step].size)),
                   'constant', constant_values = 0.)

        phi_se = np.pad(self.Phi_SE[self.max_step:end_step + max_kernel_length], 
               (0, max(0, flows_size-self.Phi_SE[self.max_step:end_step].size)),
               'constant', constant_values = 0.)
        phi_eu = np.pad(self.Phi_EU[self.max_step:end_step + max_kernel_length], 
               (0, max(0, flows_size-self.Phi_EU[self.max_step:end_step].size)),
               'constant', constant_values = 0.)
        phi_ui = np.pad(self.Phi_UI[self.max_step:end_step + max_kernel_length], 
               (0, max(0, flows_size-self.Phi_UI[self.max_step:end_step].size)),
               'constant', constant_values = 0.)
        phi_ur = np.pad(self.Phi_UR[self.max_step:end_step + max_kernel_length], 
               (0, max(0, flows_size-self.Phi_UR[self.max_step:end_step].size)),
               'constant', constant_values = 0.)
        phi_ir = np.pad(self.Phi_IR[self.max_step:end_step + max_kernel_length], 
               (0, max(0, flows_size-self.Phi_IR[self.max_step:end_step].size)),
               'constant', constant_values = 0.)
        phi_rs = np.pad(self.Phi_RS[self.max_step:end_step + max_kernel_length], 
               (0, max(0, flows_size-self.Phi_RS[self.max_step:end_step].size)),
               'constant', constant_values = 0.)
        phi_v  = np.pad(self.Phi_V[self.max_step:end_step + max_kernel_length], 
               (0, max(0, compartments_size-self.Phi_V[self.max_step:end_step].size)),
               'constant', constant_values = 0.)
                
        # Main simulation loop 

        for t in tqdm(range(0, end_step - self.max_step)):
            
            # Update distribution indices
            cur_rho_e_idx = np.searchsorted(self.rho_e_t, t, side = 'right') - 1
            cur_rho_i_idx = np.searchsorted(self.rho_i_t, t, side = 'right') - 1
            cur_rho_u_idx = np.searchsorted(self.rho_u_t, t, side = 'right') - 1
            cur_rho_r_idx = np.searchsorted(self.rho_r_t, t, side = 'right') - 1

            # Get current parameters
            cur_s     = s_array[t]
            cur_tau   = tau_array[t]
            cur_v     = v_array[t]

            # Evaluate active population
            p = s[t] + e[t] + u[t] + r[t]
            
            # Evolve contagion flow
            phi_se[t] += self.beta * cur_tau * cur_s * s[t] * u[t] * self.dt / p
            phi_v[t]  += min(s[t]-phi_se[t], cur_v) 
            
            # Propagate flows
            propagate_forward(t, end_step+max_kernel_length, phi_se[t],
                              [phi_eu], self.rho_es[cur_rho_e_idx],
                              branching_ratios = np.array([1.]))
            propagate_forward(t, end_step+max_kernel_length, phi_eu[t],
                           [phi_ui, phi_ur], self.rho_us[cur_rho_u_idx],
                           branching_ratios = np.array([self.alpha, 1. - self.alpha]))
            propagate_forward(t, end_step+max_kernel_length, phi_ui[t], 
                              [phi_ir], self.rho_is[cur_rho_i_idx],
                              branching_ratios = np.array([1.]))
            propagate_forward(t, end_step+max_kernel_length,
                              phi_ir[t]+phi_ur[t]+phi_v[t],
                              [phi_rs], self.rho_rs[cur_rho_r_idx],
                              branching_ratios = np.array([1.]))
            
            # Evolve compartments
           
            s[t+1]   = s[t] - phi_se[t] + phi_rs[t] - phi_v[t]
            e[t+1]   = e[t] + phi_se[t] - phi_eu[t]
            u[t+1]   = u[t] + phi_eu[t] - phi_ui[t] - phi_ur[t]
            i[t+1]   = i[t] + phi_ui[t] - phi_ir[t]
            r[t+1]   = r[t] + phi_ir[t] + phi_ur[t] + phi_v[t] - phi_rs[t]
            tot[t+1] = s[t+1] + e[t+1] + i[t+1] + u[t+1] + r[t+1]

        # Return compartments, total to check for consistency and flows.

        return (ts, s, e, u, i, r, tot, 
                phi_se, phi_eu, phi_ui, phi_ur, phi_ir, phi_rs, phi_v)

    def run_and_emplace(self, t_end, s, variants, vaccines):

        sim_tuple = self.run(t_end, s, variants, vaccines)
        (ts, s, e, u, i, r, tot, 
         phi_se, phi_eu, phi_ui, phi_ur, phi_ir, phi_rs, phi_v) = sim_tuple
       
        print(self.max_step)
        print(self.S[self.max_step], s[0])
        print(self.E[self.max_step], e[0])
        print(self.U[self.max_step], u[0])
        print(self.I[self.max_step], i[0])
        print(self.R[self.max_step], r[0])
        
        print(self.Phi_SE[self.max_step], phi_se[0])
        print(self.Phi_EU[self.max_step], phi_eu[0])
        print(self.Phi_UI[self.max_step], phi_ui[0])
        print(self.Phi_IR[self.max_step], phi_ir[0])
        print(self.Phi_UR[self.max_step], phi_ur[0])
        print(self.Phi_RS[self.max_step], phi_rs[0])
        print(self.Phi_V[self.max_step],  phi_v[0])
        
        # Assuming self.max_step is the last valid point on the system
        # trajectory, we need to have a final size equal to self.max_step
        # + self.S.size, or any valid compartment.
        
        print(ts.max())
        
        self.ts= np.pad(self.ts,(0, max(0, self.max_step +ts.size - self.ts.size)))
        self.ts[self.max_step+1:] = ts[1:]
        
        self.TOT = np.pad(self.TOT, 
                          (0, max(0, self.max_step + tot.size - self.TOT.size)))
        self.TOT[self.max_step+1:] = tot[1:]
        
        self.S = np.pad(self.S, (0, max(0, self.max_step + s.size - self.S.size)))
        self.E = np.pad(self.E, (0, max(0, self.max_step + e.size - self.E.size)))
        self.U = np.pad(self.U, (0, max(0, self.max_step + u.size - self.U.size)))
        self.I = np.pad(self.I, (0, max(0, self.max_step + i.size - self.I.size)))
        self.R = np.pad(self.R, (0, max(0, self.max_step + r.size - self.R.size)))

        self.S[self.max_step+1:] = s[1:]
        self.E[self.max_step+1:] = e[1:]
        self.U[self.max_step+1:] = u[1:]
        self.I[self.max_step+1:] = i[1:]
        self.R[self.max_step+1:] = r[1:]

        self.Phi_SE = np.pad(self.Phi_SE, 
                             (0,max(0,self.max_step+phi_se.size-self.Phi_SE.size)))
        self.Phi_EU = np.pad(self.Phi_EU, 
                             (0,max(0,self.max_step+phi_eu.size-self.Phi_EU.size)))
        self.Phi_UI = np.pad(self.Phi_UI, 
                             (0,max(0,self.max_step+phi_ui.size-self.Phi_UI.size)))
        self.Phi_IR = np.pad(self.Phi_IR, 
                             (0,max(0,self.max_step+phi_ir.size-self.Phi_IR.size)))
        self.Phi_UR = np.pad(self.Phi_UR, 
                             (0,max(0,self.max_step+phi_ur.size-self.Phi_UR.size)))
        self.Phi_RS = np.pad(self.Phi_RS, 
                             (0,max(0,self.max_step+phi_rs.size-self.Phi_RS.size)))
        self.Phi_V  = np.pad(self.Phi_V, 
                             (0,max(0,self.max_step+phi_v.size-self.Phi_V.size)))

        self.Phi_SE[self.max_step:] = phi_se
        self.Phi_EU[self.max_step+1:] = phi_eu[1:]
        self.Phi_UI[self.max_step+1:] = phi_ui[1:]
        self.Phi_IR[self.max_step+1:] = phi_ir[1:]
        self.Phi_UR[self.max_step+1:] = phi_ur[1:]
        self.Phi_RS[self.max_step+1:] = phi_rs[1:]
        self.Phi_V[self.max_step+1:]  = phi_v[1:]
        
        self.max_step += s.size - 1

        return (self.ts, self.S, self.E, self.U, self.I, self.R, self.TOT,
                self.Phi_SE, self.Phi_EU, self.Phi_UI, self.Phi_IR, 
                self.Phi_UR, self.Phi_RS, self.Phi_V)
           
    def fit_to_data(self, fit_period, n_periods,
                    rel_t1, rel_t2, s0, variants, vaccines, 
                    data = new_positives, alignment = 0):
    
        pts_per_day = int(1./self.dt)
    
        if (rel_t2 > fit_period) or (rel_t2 < rel_t1) or (rel_t1 > fit_period):
            raise ValueError('Error in the definition of fit boundaries.')
        
        # Generate fitting points.
    
        t0s = np.arange(0, n_periods, 1., dtype = int) * fit_period
        t1s = np.arange(0, n_periods, 1., dtype = int) * fit_period + rel_t1
        t2s = np.arange(0, n_periods, 1., dtype = int) * fit_period + rel_t2

        print(t0s)
        print(t1s)
        print(t2s) 
        
        # Prepare sociability array

        s_fit = (np.concat([t0s, np.array([np.inf])]), # Timestamps
                 np.full(t0s.size + 1, s0))            # Values

        print(s_fit[0])
        print(s_fit[1])
        
        #
        # FITTING 
        #
        # Initialize lists that will contain values of the error derivative
        # and s guesses.

        dEds = [] # Derivative
        s_sm = [] # "s secant method"
        
        # The method of secants requires two initial guesses for iteration,
        # so we must handle them manually.

        # FIRST OVERALL GUESS is 1, just baseline beta.
        s_sm.append(s_fit[1][0])

        # Update the array and simulate.
        for s_val in s_fit[1]:
            s_val = s_sm[-1]

        run_res = self.run(t2s[0], s_fit, variants = variants, vaccines = vaccines)
        (ts,s,e,u,i,r,tot,phi_se,phi_eu,phi_ui,phi_ur,phi_ir,phi_rs,phi_v) = run_res
        
        timestep0 = t0s[0]*pts_per_day
        timestep1 = t1s[0]*pts_per_day
        timestep2 = t2s[0]*pts_per_day

        # Select only the part of the simulation between t1 and t2.
        phi_pred = phi_ui[timestep1 : timestep2]

        # Then sum all the contributes of each dt for each day.        
        phi_pred = phi_pred.reshape(-1, pts_per_day).sum(axis=1)
        
        assert phi_pred.shape[0] == t2s[0] - t1s[0]

        # Compute the gap between the prediction and the data.

        gap = data[t1s[0] : t2s[0]] - phi_pred

        print(phi_pred, gap)

        # Now compute the linear response for the flow.
        # First compute the concentration of Susceptibles
        n_0 = s[timestep0] / (s[timestep0]+e[timestep0]+u[timestep0]+r[timestep0])
        
        # We also need the number of Unreported cases at t0. 
        U0 = u[timestep0]

        # We need to get the parameters of the current Unreported
        # and Exposed exit distribution.
        cur_rho_e_idx = np.searchsorted(self.rho_e_t, t0s[0], side = 'right') - 1
        cur_rho_u_idx = np.searchsorted(self.rho_u_t, t0s[0], side = 'right') - 1
        
        T_E     = self.rho_e_mus[cur_rho_e_idx]
        T_U     = self.rho_u_mus[cur_rho_u_idx]
        sigma_U = self.rho_u_sigmas[cur_rho_u_idx]

        a = T_U**2/sigma_U**2
        b = T_U/sigma_U**2

        # We can now compute the local Lyapunov exponent. 
        
        mu_   = mu(self.beta, n_0, s_sm[-1], T_U, T_E, self.alpha, a)
        
        print()
        print("MU = ",mu_)
        print()
        
        # And its susceptivity.

        dmuds_ = susceptivity(mu_, self.beta, n_0, s_sm[-1], T_U, T_E, self.alpha, a)
        
        # Generate fit timesteps.

        fit_timesteps  = np.arange(t1s[0], t2s[0], 1.)
        conv_timesteps = np.arange(0, t2s[0]-t0s[0], 1.)

        print(gap, gap.shape[0])
        print(fit_timesteps, fit_timesteps.shape[0])
    
        lrt_Udot_vals = lrt_Udot(conv_timesteps, t0s[0], mu_, dmuds_, U0, T_U)
        k_vals= k(conv_timesteps, a, b).real

        print(lrt_Udot_vals)
        print(k_vals)

        conv = np.convolve(k_vals, lrt_Udot_vals)[t1s[0]-t0s[0]:t2s[0]-t0s[0]]
        
        print(conv)

        assert 0 == 1

        # SECOND OVERALL GUESS.

        s_sm.append(s_fit[1][0]*0.9) # PLACEHOLDER

        return 
        for idx, (t0, t1, t2) in enumerate(zip(t0s, t1s, t2s)):
            print(s_sm)
            print(np.fabs(s_sm[-1]-s_sm[-2])>1e-4) 
            while (np.fabs(s_sm[-1]-s_sm[-2])>1e-4) and (np.fabs(dEds[-1])>1e-4):

                # Compute new approximation for the root.
                s_sm.append(s_sm[-1]-dEds[-1]*(s_sm[-1]-s_sm[-2])/(dEds[-1]-dEds[-2]))
                
                # Update s for the future 
                s_fit[1][idx:] = s_sm[-1]
                
        pass
   #def fit_to_data(data = new_positives, fit_period = 7, start_day = data_beg_day,
#                days = 300, dt = _DT_, beta = 1/1.2, alpha = .14, 
#                N = 886891, norm = _NORM_, s0 =  1.,
#                pars_e = pars_e_test, pars_i = pars_i_test, 
#                pars_u = pars_u_test, pars_r = pars_r_test, 
#                variants = var_test, vaccines = vacc_test,
#                return_new_positives = True, return_vaccinated = False):
#
#    global _DT_, _NORM_
#    _DT_ = dt
#    _NORM_ = norm
#
#    assert days <= data.shape[-1]
#    # Calculate number of iterations
#    max_step = int(np.rint(days / dt))
#    
#    # Create fit interval limits
#    fit_limits = np.array([i * fit_period for i in range(int(days//fit_period)+1)])
#    fit_pairs  = zip(fit_limits[:-1], fit_limits[1:])
#    
#    # Main simulation loop
#    
#
#    t = np.array([t for t in range(max_step+1)])
#    if return_new_positives:
#        if return_vaccinated:
#            return (t, S, E, I, U, R, Phi_UI, Phi_V, TOT)
#        else:
#            return (t, S, E, I, U, R, Phi_UI, TOT)
#    else:
#        if return_vaccinated:
#            return(t, S, E, I, U, R, Phi_V, TOT)
#        else:
#            return (t, S, E, I, U, R, TOT)

def test_model(days = 200, dt = _DT_, norm = False):
    print("Simulate", days, "days with a {:.2f}".format(dt), "day resolution.")

    s_test = (np.array([0, 40, 60, np.inf]), np.array([1., .15, .15, .5]))

    m = model(dt = dt, 
              beta = 1./1.2, 
              alpha = .14, 
              N = 886891, 
              norm = norm, 
              dist_e = dist_e_test, 
              dist_i = dist_i_test, 
              dist_u = dist_u_test, 
              dist_r = dist_r_test)
    
    m.fit_to_data(7, 20, 2, 7,  s0 = 1.0, variants = var_test, 
                  vaccines = vacc_test, data = new_positives)

    _   = m.run_and_emplace(days//4, s = s_test, variants = var_test, vaccines = vacc_test)
    _   = m.run_and_emplace(days//2, s = s_test, variants = var_test, vaccines = vacc_test)
    res = m.run_and_emplace(days, s = s_test, variants = var_test, vaccines = vacc_test)
    t,s,e,u,i,r,tot,phi_se,phi_eu,phi_ui,phi_ur,phi_ir,phi_rs,phi_v = res
    print(t.size, t.max()/dt)
#Test flow reconstruction

    kernel = k(t, alpha, beta).real
    
    du = u[1:] - u[:-1]
    
    conv = np.convolve(kernel, du[:t.shape[0]//3])[:t.shape[0]//3] * 0.14 * dt

#%% Graphics
    
    plt.rcParams["figure.autolayout"] = True
     
    fig, ax = plt.subplots(2,1, figsize = (12,8), sharex = True)

    ax[0].plot(t, s, label = 'S')
    ax[0].plot(t, e, label = 'E')
    ax[0].plot(t, i, label = 'I')
    ax[0].plot(t, u, label = 'U')
    ax[0].plot(t, r, label = 'R')
    ax[0].plot(t, tot, label = 'R')
    
    newp = phi_ui[:t.size]

    ax[1].plot(t, newp, label = 'New positive cases')    
    ax[1].plot(t[:t.size//3], conv, label = r'$a \ast \dot{U}(t)$ np.conv', color = 'red',
               linestyle = "dashed") 
    ax[1].set_xlabel('Days since the beginning of the epidemic', fontsize = 14)
    ax[1].set_ylabel('Individuals', fontsize = 14)
    ax[1].set_ylim(bottom = 0)

    ax[0].legend(fontsize = 14)
    ax[1].legend(fontsize = 14, loc = "upper left")
    plt.show()
    #df_test = pd.read_csv('../effective/out.csv')
     
    #ax[0].scatter(df_test['t'], df_test.S, label = 'S', marker='+', color='C0')
    #ax[0].scatter(df_test['t'], df_test.E, label = 'E', marker='+', color='C1')
    #ax[0].scatter(df_test['t'], df_test.I, label = 'I', marker='+', color='C2')
    #ax[0].scatter(df_test['t'], df_test.U, label = 'U', marker='+', color='C3')
    #ax[0].scatter(df_test['t'], df_test.R, label = 'R', marker='+', color='C4')
    
    fig2, ax2 = plt.subplots(figsize = (12,8))

    ax2.plot([idx*dt for idx in range(len(phi_se))], phi_se,    label = r'$\phi_{SE}$')
    ax2.plot([idx*dt for idx in range(len(phi_eu))], phi_eu,    label = r'$\phi_{EU}$')
    ax2.plot([idx*dt for idx in range(len(phi_ui))], phi_ui,    label = r'$\phi_{UI}$')
    ax2.plot([idx*dt for idx in range(len(phi_ir))], phi_ir,    label = r'$\phi_{IR}$')
    ax2.plot([idx*dt for idx in range(len(phi_ur))], phi_ur,    label = r'$\phi_{UR}$')
    ax2.plot([idx*dt for idx in range(len(phi_rs))], phi_rs,    label = r'$\phi_{RS}$')
    ax2.legend()

    if norm:
        ax[0].set_ylim(bottom = 0, top = 0.0025)
        ax[0].set_ylabel('Population Fraction', fontsize = 14)
    else:
        ax[0].set_ylim(bottom = 0)#, top = 250000)
        ax[0].set_ylabel('Individuals', fontsize = 14)

    ax[0].set_xlim([0, max(t)])

    #ax[0].vlines(var_test[0], *ax[0].get_ylim(), linestyle = 'dashed',
    #             color = 'red', label = 'Variant arrival')
    #ax[0].vlines(vacc_test[0], *ax[0].get_ylim(), linestyle = 'dashed',
    #             color = 'blue', label = 'Beginning of vaccination')


if __name__ == "__main__":
    test_model(dt = _DT_, norm = False)
