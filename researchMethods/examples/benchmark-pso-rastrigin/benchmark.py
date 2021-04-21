#!/usr/bin/env python
import time
import sys

import numpy as np

from fffit import pso

def rastrigin(x):
    """Rastrigin test function"""
    return np.sum(x*x - 10 * np.cos(2*np.pi*x) + 10)


def main():
    np.set_printoptions(suppress=True, precision=4, linewidth=150)
    popsize = 100
    ranges = 10.12
    ndim = 30
    fit = []
    times = []
    bounds = np.array((np.ones(ndim) * ranges * -1, np.ones(ndim) * ranges))
    w = 0.728994
    c = (0.5, 2.0)

    for run in range(20):
        print(f"Starting run {run}...", file=sys.stderr)
        start_time = time.perf_counter()
        p = pso.PSO(maxiter=100, goal=1, w=w, c=c)
        p.ncpu = 1
        p.populate(popsize, x0=np.zeros(ndim), bounds=bounds, sigma=1)
        start_time = time.perf_counter()
        p.run(rastrigin, enum_particles=False, add_step_num=False, PSO_DEBUG='surf.dat')
        stop_time = time.perf_counter()
        with open("best.dat", "a") as b:
            np.savetxt(b, np.hstack([p.pos_best_glob, p.fitness]))
        with open("best_sqradius.dat", "a") as b:
            np.savetxt(b, np.hstack([np.dot(p.pos_best_glob, p.pos_best_glob), p.fitness]))
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
