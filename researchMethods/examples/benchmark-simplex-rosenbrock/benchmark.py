#!/usr/bin/env python
from fffit import simplex
import numpy as np

def rosen(x):
    """The Rosenbrock function."""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

x0 = 2*np.random.rand(30)
# simplex = Simplex(rosen, x0, maxiter=10e+7, maxfev=10e+10, xatol=1e-8, disp=True, fatol=None, adaptive=True)
simplex = simplex.Simplex(rosen, x0, maxiter=10e+7, maxfev=10e+10)
result_one_step = simplex.do_full_step()
result_complet = simplex.run(rosen,enum_particles=True, add_step_num=True, DEBUG='surface.dat')
print('result_one_step: ', result_one_step)
print('result_complet: ', result_complet)