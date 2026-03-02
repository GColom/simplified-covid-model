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

# GLOBAL DEFAULTS

_DT_ = 1./48.

# Experimental data import

df = pd.read_csv('data/COVID_data.csv',
                index_col = 'datetime', 
                parse_dates = True, date_format = "%d/%m/%Y")

df.index = pd.to_datetime(df.index)
data_beg_date = df.index.min()
data_beg_day  = df.days.min()

print("Data begins on", data_beg_date, ', corresponding to day n.', data_beg_day)

new_positives = (df['new_positives'].rolling(7, center = False, min_periods = 1)
                                    .mean().to_numpy())[3:366]

from matplotlib import pyplot as plt

# Transcendental equation that determines the Local Lyapunov exponent (LLE) mu.
def eq_mu(x, beta, n_0, s, T_u, T_e, a):
    ret = ((1-(a**a)/(x+a)**a)*np.exp(-T_e*x/T_u) - x/(T_u*n_0*beta*s))
    return ret.real 

# Auxiliary function in the calculation of the LLE susceptivity.
def Fprime(x, beta, n_0, s, T_u, T_e, a):
    return np.exp(-T_e/T_u * x) * ((a/(x+a))**a * ( a/(x+a) + T_e/T_u) - T_e/T_u)

def eq_prime(x, beta, n_0, s, T_u, T_e, a):
    ret = Fprime(x, beta, n_0, s, T_u, T_e, a) - 1/(T_u*s*beta*n_0)
    return ret.real

# Root finding for the LLE.
def mu(beta, n_0, s, T_u, T_e, a):
    tiny = 1e-12 # Exclude the root in 0, and stay out of its attraction basin.
    try:
        ret = root_scalar(eq_mu, bracket = (-5, -tiny), method = 'bisect',
                    args=(beta, n_0, s, T_u, T_e, a))
    except ValueError:
        ret = root_scalar(eq_mu, bracket = (tiny, 10.), method = 'bisect',
                    args=(beta, n_0, s, T_u, T_e, a))
    return ret.root

# Susceptivity computation.
def susceptivity(x, beta, n_0, s, T_u, T_e, a):
    return x/(s - T_u * beta * n_0 * s**2 * Fprime(x, beta, n_0, s, T_u, T_e, a))

def k(t, a, b):
    return b**a * t**(a - 1) * mittag_leffler(b**a * t**a, a, a)*np.exp(-b*t)

def lrt_Udot(t, t0, mu, dmuds, U0, T_U):
    kappa = U0*dmuds/T_U
    sigma = mu/T_U
    return kappa*np.exp(sigma*(t-t0))*(sigma*(t-t0) + 1)

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

s_assumption = lambda idx : idx if idx >= 0 else 0

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

        self.N           = N
        self.dt          = dt 
        self.pts_per_day = int(1/dt)
        print("init with dt =", self.dt)
        self.beta        = beta
        print("init with beta =", self.beta)
        self.alpha       = alpha
        self.norm        = norm
        self.dist_e      = dist_e
        self.dist_i      = dist_i
        self.dist_u      = dist_u
        self.dist_r      = dist_r
 
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

        s_array= np.array([s_vals[
            s_assumption(np.searchsorted(s_t, t, side = 'right') - 1)] 
                           for t in range(self.max_step, end_step+1)])
        plt.figure()
        plt.plot(np.arange(0., t_end, self.dt), s_array[:-1])
        
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
       
        # Assuming self.max_step is the last valid point on the system
        # trajectory, we need to have a final size equal to self.max_step
        # + self.S.size, or any valid compartment.
        
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
           
    def __gradient(self, t0, t1, t2, s_guess, variants, vaccines,
                   data = new_positives, alignment = 0):

        # Run simulation with the passed parameters 
        run_res = self.run(t2, s_guess, variants = variants, vaccines = vaccines)
        (ts,s,e,u,i,r,tot,phi_se,phi_eu,phi_ui,phi_ur,phi_ir,phi_rs,phi_v) = run_res
        
        timestep0 = t0 * self.pts_per_day
        timestep1 = t1 * self.pts_per_day
        timestep2 = t2 * self.pts_per_day

        # Select only the part of the simulation between t1 and t2.
        phi_pred = phi_ui[timestep1 : timestep2]# : self.pts_per_day]

        # Then sum all the contributes of each dt for each day.        
        phi_pred = phi_pred.reshape(-1, self.pts_per_day).mean(axis=1)
        
        assert phi_pred.shape[0] == (t2 - t1)

        # Compute the gap between the prediction and the data.

        gap = phi_pred - data[t1 : t2]

        print(s_guess[0])
        print(s_guess[1])
        print(gap)
        
        plt.figure()
        plt.plot(new_positives)
        plt.plot(ts, phi_ui[:ts.shape[0]])
        plt.plot(np.arange(t1, t2, 1.), gap)
        plt.xlim(0, 200)
        plt.twinx()
        plt.vlines(s_guess[0], ymin = 0., ymax = 1.5, linewidth = 0.5,
                   linestyle = 'dashed', color = 'C1')
        plt.vlines(t0, ymin = 0., ymax = 1.5, linewidth = 1.5,
                   linestyle = 'dashed', color = 'C1')
        plt.vlines(t1, ymin = 0., ymax = 1.5, linewidth = 1.5,
                   linestyle = 'dashed', color = 'green')
        plt.vlines(t2, ymin = 0., ymax = 1.5, linewidth = 1.5,
                   linestyle = 'dashed', color = 'red')
        plt.plot(s_guess[0], s_guess[1])
        plt.ylim(0., 1.5)
        plt.show()

        # Now compute the linear response for the flow.
        # First compute the concentration of Susceptibles

        n_0 = s[timestep0] / (s[timestep0]+e[timestep0]+u[timestep0]+r[timestep0])
        
        # We also need the number of Unreported cases at t0. 
        U0 = u[timestep0]

        # We need to get the parameters of the current Unreported
        # and Exposed exit distribution. Use t0 as rho_*_t are indexed
        # in day units.
        # We also get the current scalar sociability value cur_s.

        cur_rho_e_idx = np.searchsorted(self.rho_e_t, t0, side = 'right') - 1
        cur_rho_u_idx = np.searchsorted(self.rho_u_t, t0, side = 'right') - 1
        cur_s_idx     = s_assumption(np.searchsorted(s_guess[0], t0, side = 'right') - 1)
        
        T_E     = self.rho_e_mus[cur_rho_e_idx]    * self.dt
        T_U     = self.rho_u_mus[cur_rho_u_idx]    * self.dt
        sigma_U = self.rho_u_sigmas[cur_rho_u_idx] * self.dt
        cur_s   = s_guess[1][cur_s_idx]
        
        # These are the parameters of the U gamma distribution.

        a = T_U**2/sigma_U**2
        b = T_U/sigma_U**2

        # We can now compute the local Lyapunov exponent. 
        
        mu_   = mu(self.beta, n_0, cur_s, T_U, T_E, a)
    
        print(r'\mu =', mu_)
        
        # And its susceptivity.

        dmuds_ = susceptivity(mu_, self.beta, n_0, cur_s, T_U, T_E, a)

        print(r'\dv{\mu}{s} =', dmuds_)
        
        # Generate fit timesteps in units of days and 
        # convolution timesteps in units of dt.

        conv_timesteps = np.arange(t1, t2, self.dt)
    
        lrt_Udot_vals = lrt_Udot(np.arange(t0, t2, self.dt), t0, mu_, dmuds_, U0, T_U)
        #plt.figure()
        #plt.plot(np.arange(t0, t2, self.dt), lrt_Udot_vals)
        #plt.show()

        # Include the symptomatic fraction in the kernel.
        k_vals = self.alpha * k(conv_timesteps, a, b).real

        conv = np.convolve(k_vals,lrt_Udot_vals)[timestep1-timestep0:timestep2-timestep0]
        
        # NOW WE CAN COMPUTE THE ERROR GRADIENT
        # FIRST RESAMPLE THE CONVOLUTION EACH pts_per_day

        conv = conv[::self.pts_per_day]
        
        # dEds is just the scalar product of conv with gap.
        
        print('gap.shape = ', gap.shape)
        print('conv.shape = ', conv.shape)
        
        gradient = np.dot(gap, conv)/(t2-t1)

        print("Gradient is ", gradient)

        return gradient, run_res

    def fit_to_data(self, fit_period, n_periods,
                    rel_t1, rel_t2, s0, s1, variants, vaccines, 
                    data = new_positives, alignment = 0,
                    gtol = 1e-3, dgtol = 1e-3, dstol = 1e-3):
    
        if (rel_t2 > fit_period) or (rel_t2 < rel_t1) or (rel_t1 > fit_period):
            raise ValueError('Error in the definition of fit boundaries.')
        
        # Generate fitting points.
        t0s = alignment+np.arange(0, n_periods, 1., dtype = int) * fit_period - rel_t1
        t1s = alignment+np.arange(0, n_periods, 1., dtype = int) * fit_period
        t2s = alignment+np.arange(1, n_periods + 1, 1., dtype = int) * fit_period #+ rel_t2
        t1s = np.array([30, 45, 75, 100]) 
        t0s = t1s - rel_t1
        t2s = np.concatenate((t1s[1:], np.array([120])))

        print(t0s)
        print(t1s)
        print(t2s) 
        
        # Prepare sociability array

        s_fit = (np.concat([t0s, np.array([np.inf])]), # Timestamps
                 np.full(t0s.size + 1, s0))            # Values

        print(s_fit[0])
        print(s_fit[1])
        
        # FITTING 

        # Initialize lists that will contain values of the error derivative
        # and s guesses.

        dEds = [] # Derivative
        s_rf = [] # "s root finding"
        
        # Derivative-free methods need two initial guesses.

        # First overall guess is s0, just baseline beta.
        s_rf.append(s0)

        dEds.append(self.__gradient(t0s[0],t1s[0],t2s[0], s_fit, variants, vaccines,
                    data = new_positives, alignment = 40)[0])

        # Second overall guess.

        s_rf.append(s1)

        # Update the array and simulate.

        s_fit[1][0:] = s_rf[-1]
        
        dEds.append(self.__gradient(t0s[0],t1s[0],t2s[0], s_fit, variants, vaccines,
                    data = new_positives, alignment = 40)[0])

        for idx, (t0, t1, t2) in enumerate(zip(t0s, t1s, t2s)):

            print(f'Minimizing over [{t1}, {t2}] ({idx}-th interval), b.p. in {t0}.')
            run_condition =   ((abs(dEds[-1]) > gtol)
                           and (abs(dEds[-1] - dEds[-2]) > dgtol)
                           and (abs(s_rf[-1] - s_rf[-2]) > dstol))
            if idx != 0:
                run_condition = True

            forced_iterations = 3
            lidx = 0

            while run_condition or (lidx <= forced_iterations):
                lidx += 1

                # Compute new approximation for the root.

                # Boosted bisection
                #next_s = s_rf[-1] - dEds[

                # Secant method
                next_s = s_rf[-1]-dEds[-1]*(s_rf[-1]-s_rf[-2])/(dEds[-1]-dEds[-2])
                s_rf.append(next_s if next_s > 0. else 5e-2)

                # Update the array and simulate.
                s_fit[1][idx:] = s_rf[-1]

                dEds.append(self.__gradient(t0, t1, t2, s_fit, variants, 
                            vaccines, data = new_positives, alignment = 40)[0])

                run_condition =((abs(dEds[-1]) > gtol)
                            and (abs(dEds[-1] - dEds[-2]) > dgtol)
                            and (abs(s_rf[-1] - s_rf[-2]) > dstol))

                print('Continuation conditions are:')
                print(f'abs(dEds[-1]) ({abs(dEds[-1]):.3f}) > gtol ({gtol:.3f}):', 
                      abs(dEds[-1]) > gtol)
                print(f'abs(dEds[-1] - dEds[-2]) ({(abs(dEds[-1] - dEds[-2])):.3f}) > dgtol ({dgtol:.3f}):', 
                            abs(dEds[-1] - dEds[-2]) > dgtol)
                print(f'abs(s_rf[-1] - s_rf[-2]) ({abs(s_rf[-1] - s_rf[-2]):.3f})> dstol ({dstol:.3f}):', 
                            abs(s_rf[-1] - s_rf[-2]) > dstol)
                print('idx =', idx)
                print('lidx =', lidx)
            print("****Proceeding to new time interval.****")

        return 0 

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
    
    m.fit_to_data(fit_period = 8, n_periods = 10, rel_t1 = 2, rel_t2 = 8,  
                  s0 = 1.1, s1 = 0.9, variants = var_test, 
                  vaccines = vacc_test, data = new_positives, alignment = 27)

if __name__ == "__main__":
    test_model(dt = _DT_, norm = False)
