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

# NdC Stesse definizioni di alpha e beta ma per evitare conflitti meglio rinominare
# tutto.

a       = T_u**2/sigma_u**2
b       = T_u/sigma_u**2

print("Run with "+chr(945)+f" = {alpha:.3f}, "+chr(946)+f" = {beta:.3f}.")
print(f"Gamma mean value T_U = {T_U:.1f}, 1/T_U = {1/T_U:.4f}.")

def b(t, alpha, beta):
    return beta**alpha * t**(alpha - 1) * mittag_leffler(beta**alpha * t**alpha, 
                                                         alpha, alpha)*np.exp(-beta*t)
def gamma_distrib(t, alpha, beta):
    return beta**alpha * t**(alpha - 1) * np.exp(-beta*t) / gamma(alpha)

def lrt_Udot(t, t0, mu, dmuds, U0, T_U):
    kappa = U0*dmuds/T_U
    sigma = mu/T_U
    return kappa*np.exp(sigma*(t-t0))*(sigma*(t-t0) +1)

# Transcendental equation that determines the Local Lyapunov exponent (LLE) mu.
def eq_mu(x, beta, n_0, m, T_u, T_e, alpha, a):
    return (1-(a**a)/(x+a)**a)*np.exp(-T_e*x/T_u) - x/(T_u*n_0*beta*m)

# Auxiliary function in the calculation of the LLE susceptivity.
def Fprime(x, beta, n_0, m, T_u, T_e, alpha, a):
    return np.exp(-T_e/T_u * x) * ((a/(x+a))**a * ( a/(x+a) + T_e/T_u) - T_e/T_u)

def mu(beta, n_0, s, T_u, T_e, alpha, a):
    try:
        ret = root_scalar(eq_mu, bracket = (- 10, -1e-6, ), method = 'bisect',
                    args=(beta, n_0, s, T_u, T_e, alpha, a))
    except ValueError:
        ret = root_scalar(eq_mu, bracket = (1e-6, 10., ), method = 'bisect',
                    args=(beta, n_0, s, T_u, T_e, alpha, a))
    return ret.root

def susceptivity(x, beta, n_0, s, T_u, T_e, alpha, a):
    return x/(s - T_u * beta * n_0 * s**2 * Fprime(x, beta, n_0, s, T_u, T_e, alpha, a))

ts   = np.linspace(0, 100, 5000)
mlvs = b(ts, alpha, beta).real

plt.plot(ts, mlvs, zorder = 0, linewidth = 2, 
         linestyle = 'dashed', label = r'$b(t, \alpha, \beta)$')

gvs = np.zeros_like(ts)

for k in range(1, 9):
    gvs += gamma_distrib(ts, k * alpha, beta)
    plt.plot(ts, gvs, zorder = -1, label = fr'$b_{k}(t, \alpha, \beta)$')

plt.xlim(ts.min(), ts.max())
plt.ylim(0., 1.1*mlvs.max())
plt.xlabel('$t$', fontsize = 14)
#plt.hlines(1/T_U, ts.min(), ts.max())
plt.legend(fontsize = 12, ncols = 2)
plt.tight_layout()

plt.figure()
plt.plot(ts, np.abs(mlvs - 1/T_U), zorder = 0, linewidth = 2, 
         label = r'$|b(t, \alpha, \beta)-\beta|$')

plt.figure()

sigma = 1.e-3
kappa = 0.2

#ts = np.linspace(0, 100, 1000)

lrs = lrt(ts, 0, sigma, kappa)

conv = np.convolve(mlvs, lrs)[:ts.shape[0]]

plt.plot(ts, conv)

plt.show()
