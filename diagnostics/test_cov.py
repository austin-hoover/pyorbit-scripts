"""Compute bunch covariance matrix."""
import os
import sys

import numpy as np

from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis

from orbit_tools.bunch import get_bunch_cov


# Generate a normal distribution. Note: since ORBIT accounts for dispersion, x-dE 
# (0-5) or y-dE (2-5) correlations will change the computed Twiss parameters.
rng = np.random.default_rng(15520)

cov = np.identity(6)
for (i, j) in [(0, 1), (2, 3), (4, 5), (0, 2)]:
    cov[i, j] = cov[j, i] = rng.uniform(-0.8, 0.8)
cov *= 100.0

points = rng.multivariate_normal(np.zeros(6), cov, size=100_000)

bunch = Bunch()
for (x, xp, y, yp, z, de) in points:
    bunch.addParticle(x, xp, y, yp, z, de)


cov_numpy = np.cov(points.T)
cov_orbit = get_bunch_cov(bunch)

for i in range(6):
    for j in range(i + 1):
        print(f"cov[{i}, {j}] (numpy) = {cov_numpy[i, j]}")
        print(f"cov[{i}, {j}] (orbit) = {cov_orbit[i, j]}")
