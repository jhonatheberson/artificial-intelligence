#!/usr/bin/env python
import time
import sys

import numpy as np

from fffit import ga
from scipy.optimize import rosen

def sphere(x):
    total = 0
    return -np.dot(x,x)

def rastrigin(x):
    total = 0
    return -np.sum((x)**2 - 10 * np.cos(2*np.pi*x) + 10)


def rosenbrock(x):
    """ Rosenbrock test function

    As implemented in scipy."""
    return -rosen(x)


def main():
    np.set_printoptions(suppress=True, precision=4, linewidth=150)
    popsize = 100
    ranges = 100
    ndim = 30
    fit = []
    times = []
    step = []
    bounds = np.array((np.ones(ndim) * ranges * -1, np.ones(ndim) * ranges))
    w = 0.728994
    c = (0.5, 2.5)

    for run in range(20):
        print(f"Starting run {run}...", file=sys.stderr)
        start_time = time.perf_counter()
        p = ga.Genetic(maxiter=1000, goal=0, cross_over='two_points',
                       mutation_probability=0.01, mutation='uniform',
                       selection_method='elitism',num_parents=2,
                       num_elitism=10, bounds=bounds)
        p.ncpu = 1
        p.populate(popsize, x0=np.zeros(ndim), bounds=bounds)
        start_time = time.perf_counter()
        p.run(rastrigin, enum_particles=False, add_step_num=False, DEBUG=f'fitness_sphere_{run}.dat')
        stop_time = time.perf_counter()
        step.append(p.step_number)
        fit.append(p.fitness)
        times.append(stop_time - start_time)

    stepavg = np.mean(step)
    stepstdev = np.std(step)
    fitavg = np.mean(fit)
    fitstdev = np.std(fit)
    timeavg = np.mean(times)
    timestdev = np.std(times)

    print(f"Fitness: {fitavg:e} ± {fitstdev:e}")
    print(f"step: {stepavg:e} ± {stepstdev:e}")
    print(f"Time: {timeavg} ± {timestdev}")
    print(f"troca de population: {p.cont_new_population}")


if __name__ == '__main__':
    main()
