#!/usr/bin/env python
import time
import sys

import numpy as np

from fffit import pso
from scipy.optimize import rosen


def rosenbrock(x):
    """ Rosenbrock test function

    As implemented in scipy."""
    return rosen(x)


def main():
    np.set_printoptions(suppress=True, precision=4, linewidth=150)
    popsize = 100
    ranges = 100
    ndim = 2
    fit = []
    times = []
    bounds = np.array((np.ones(ndim) * ranges * -1, np.ones(ndim) * ranges))
    w = 0.728994
    c = (0.5, 2.5)

    for run in range(20):
        print(f"Starting run {run}...", file=sys.stderr)
        start_time = time.perf_counter()
        p = pso.PSO(maxiter=100, goal=0, w=w, c=c)
        p.ncpu = 1
        p.populate(popsize, x0=np.zeros(ndim), bounds=bounds, sigma=1)
        start_time = time.perf_counter()
        p.run(rosenbrock, enum_particles=False, add_step_num=False, PSO_DEBUG='surface.dat')
        stop_time = time.perf_counter()
        fit.append(p.fitness)
        times.append(stop_time - start_time)

    fitavg = np.mean(fit)
    fitstdev = np.std(fit)
    timeavg = np.mean(times)
    timestdev = np.std(times)

    print(f"Fitness: {fitavg} ± {fitstdev}")
    print(f"Time: {timeavg} ± {timestdev}")


if __name__ == '__main__':
    main()
