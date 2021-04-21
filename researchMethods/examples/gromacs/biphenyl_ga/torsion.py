#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import time

from pathlib import Path

import numpy as np
import panedr

from fffit import ga
from fffit.helpers import gromacs

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_particles', dest='num_particles',
                        action='extend', nargs='+', type=int,
                        help='Number of particles in the swarm', required=True)
    parser.add_argument('-r', '--runs', dest='runs', type=int,
                        help='Number of times to repeat the run', required=True)
    parser.add_argument('-p', '--ncpu', dest='ncpu', type=int,
                        help='Number of processors', default=1)
    parser.add_argument('-s', '--sourcedir', dest='sourcedir', type=Path,
                        help='Source directory: read-only, location of gro, top, mdp files.',
                        default=Path.cwd())
    return parser.parse_args()

def main():
    args = parseargs()
    tarprocs = []
    tarprocidx = []
    angles = np.linspace(-90, 85, 36, dtype=int)
    reference_dir = args.sourcedir
    reference = np.loadtxt(reference_dir.joinpath('reference.dat'))[:,1]
    basename = 'biphenyl'
    topfile = reference_dir.joinpath('top', 'biphenyl.top')
    mdpfile = reference_dir.joinpath('mdp', 'em.mdp')
    grodir = reference_dir.joinpath('gro')
    basedir=reference_dir
    ndim = 6
    x0 = np.zeros(ndim)
    results = {}
    

    
    for popsize in args.num_particles:
        results[popsize] = []
        for attempt in range(args.runs):
            pes = gromacs.PES(reference, points=angles, basedir=f'{popsize}_{attempt}',
                              basename=basename, topfile=topfile,
                              mdpfile=mdpfile, grodir=grodir)
            start_time = time.perf_counter()
            bounds = np.array((np.ones(ndim) * -100, np.ones(ndim) * 100))
            P = ga.Genetic(maxiter=1000, goal=0, cross_over='one_point',
                       mutation_probability=0.01, mutation='uniform',
                       selection_method='elitism',num_parents=2,
                       num_elitism=10, bounds=bounds)
            P.ncpu = 4
            P.populate(popsize, x0=np.zeros(ndim), bounds=bounds)
            P.run(pes.fitness, enum_particles=True, add_step_num=True, DEBUG=f'surface_{popsize}.dat')
            # P.run(pes.fitness, enum_particles=True,
            #       add_step_num=True, PSO_DEBUG='surface.dat')
            stop_time = time.perf_counter()
            delta = stop_time - start_time
            results[popsize].append({'nsteps': P.step_number,
                                     'time': delta,
                                     'fitness': P.fitness})
            with open(f"results_{popsize}.dat", "a") as res:
                print(popsize, P.step_number, P.fitness, delta, file=res)
            print(f'Compressing {popsize}_{attempt}...')
            tarprocs.append(subprocess.Popen(['tar', '-zcf',
                                              f'{popsize:03d}_{attempt:02d}.tar.gz',
                                              'run',f'{popsize:03d}_{attempt:02d}'
                                             ]))
            for procidx, proc in enumerate(tarprocs):
                if proc.poll() == 0:
                    tarprocidx.append(procidx)
                    shutil.rmtree(proc.args[3], ignore_errors=True)
        with open(f'results_{popsize}.dat', "a") as res:
            print("\n\n", file=res)
    with open("dump.txt", "w") as dump:
        print(results, file=dump)

if __name__ == '__main__':
    main()
