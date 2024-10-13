#!/usr/bin/env python3

import sys
MIN_PYTHON = (3, 7)
assert sys.version_info >= MIN_PYTHON

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, uniform

M = 1.0
η = 1.0
Δ = 0.01/M

lawV = norm(loc=0, scale=1/M)
lawU = uniform(loc=0, scale=1)

N = 1000000
Z = lawV.rvs(size=N) - η/2
U = lawU.rvs(size=N)
X = 0.5 * ((Z - η/2) + np.sqrt((Z + η/2) ** 2 - 2 * (1/M) ** 2 * np.log(U)))
W = Z[X <= η/2]

bins = np.arange(-6/M, 6/M, Δ)
hist = np.histogram(W, bins=bins, density=True,)
w, pw = hist[1][:-1], hist[0]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3,), constrained_layout=True,)
ax.plot(w, pw, '-k', lw=0.5,)
ax.axvline(x=η/2, ymin=-10.0/Δ, ymax=+10.0/Δ, color='k', lw=0.5,)
ax.set_xlabel(r'$w$')
ax.set_ylabel(r'$\phi(w)$')
plt.show()
