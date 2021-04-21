#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import time

from pathlib import Path

import numpy as np
import panedr

from fffit import pso
from fffit.helpers import gromacs

def main():
    tarprocs = []
    tarprocidx = []
    angles = np.linspace(-90, 85, 36, dtype=int)
    reference = np.loadtxt('reference.dat')[:,1]
    basename = 'biphenyl'
    topfile = Path.cwd().joinpath('top', 'biphenyl.top')
    mdpfile = Path.cwd().joinpath('mdp', 'em.mdp')
    grodir = Path.cwd().joinpath('gro')
    basedir=Path.cwd()
    ndim = 2
    w = 0.728994
    c = (0.5, 2.5)

    x0 = np.zeros(ndim)
    results = {}

    for popsize in np.linspace(10, 100, 11, dtype=int):
        results[popsize] = []
        for attempt in range(10):
            curdir=Path.cwd().joinpath(f'{popsize}_{attempt}')
            pes = gromacs.PES(reference, points=angles, basedir=curdir,
                              basename=basename, topfile=topfile,
                              mdpfile=mdpfile, grodir=grodir)
            start_time = time.perf_counter()
            bounds = np.array((np.ones(ndim) * -100, np.ones(ndim) * 100))
            P = pso.PSO(maxiter=100, goal=1, w=w, c=c)
            P.ncpu = 8
            P.populate(popsize, x0=x0, bounds=bounds, sigma=2)
            P.run(pes.fitness, enum_particles=True,
                  add_step_num=True, PSO_DEBUG='surface.dat')
            stop_time = time.perf_counter()
            delta = stop_time - start_time
            results[popsize].append({'nsteps': P.step_number,
                                     'time': delta,
                                     'fitness': P.fitness,
                                     'pos_best': P.pos_best_glob})
            with open("results.dat", "a") as res:
                print(popsize, P.step_number, P.fitness, delta, file=res)
            print(f'Compressing {popsize}_{attempt}...')
            tarprocs.append(subprocess.Popen(['tar', '-Jcf',
                                             f'{popsize}_{attempt}.tar.xz',
                                             f'{popsize}_{attempt}']))
            for procidx, proc in enumerate(tarprocs):
                if proc.poll() == 0:
                    tarprocidx.append(procidx)
                    shutil.rmtree(proc.args[3], ignore_errors=True)
        with open("results.dat", "a") as res:
            print("\n\n", file=res)
    with open("dump.txt", "w") as dump:
        print(results, file=dump)

if __name__ == '__main__':
    main()
