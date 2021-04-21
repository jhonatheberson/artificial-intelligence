#!/usr/bin/python3.6
import argparse
import pickle
import os.path
import logging
from confparse import read_conf

import numpy as np

from fffit import pso


def rosenbrock(x):
    total = 0
    for i in range(x.size - 1):
        total += 100 * ((x[i] ** 2 - x[i + 1]) ** 2) + (1 - x[i]) ** 2
    return total
# END


def createPso(fitness_tests, num_particles, maxiter, initial, bounds):
    P = pso.PSO(fitness_tests, maxiter=maxiter)
    P.populate(num_particles, initial, bounds)
    return P


def write(PSO, fileName):
    with open(fileName, 'wb') as p:
        pickle.dump(PSO, p)


def read(fileName):
    with open(fileName, 'rb') as p:
        T = pickle.load(p)
    return T


def swarmUpdate(PSO, bounds):
    PSO.swarmUpdate(bounds)


def updateFitness(PSO, function):
    return PSO.calculateFitness(function)


def initialize(fitness_tests, num_particles, maxiter, initial, bounds,
               writeFile):
    P = createPso(fitness_tests, num_particles, maxiter, initial, bounds)
    write(P, writeFile)


def prepareJobs():
    # TODO
    pass


def fitness(readFile, writeFile):
    if (os.path.exists(readFile)):
        T = read(readFile)
        fitness = updateFitness(T, rosenbrock)
        print(fitness)
        write(T, writeFile)
    else:
        logging.debug("file does not exist with this name.")


def step(bounds, readFile, writeFile):
    if (os.path.exists(readFile)):
        T = read(readFile)
        swarmUpdate(T, bounds)
        print([p.position for p in T.swarm])
        write(T, writeFile)
    else:
        logging.debug("file does not exist with this name.")


def parse_args():
    parser = argparse.ArgumentParser()
    runmode = parser.add_mutually_exclusive_group()
    parser.add_argument('-c', '--conffile',
                        default='conf.ini',
                        help='Read configuration file',
                        required=True)
    runmode.add_argument('-i', '--init',
                         help='Create PSO object',
                         action='store_true')
    runmode.add_argument('-p', '--prepare',
                         help='Prepare and submit fitness jobs.',
                         action='store_true')
    runmode.add_argument('-f', '--fitness',
                         help='calculates the fitnnes',
                         action='store_true')
    runmode.add_argument('-s', '--steps',
                         help='calculates the step, updating' +
                              ' speeds and positions',
                         action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    # PSO_setup, test_setup = read_conf(args.conffile)
    PSO_setup = read_conf(args.conffile)

    if args.init:
        initialize(PSO_setup['fitness_tests'],
                   PSO_setup['num_particles'],
                   PSO_setup['maxiter'],
                   PSO_setup['initial'],
                   PSO_setup['bounds'],
                   PSO_setup['WriteFile'])

    elif args.prepare:
        prepareJobs()

    elif args.fitness:
        fitness(PSO_setup['WriteFile'], PSO_setup['ReadFile'])

    elif args.steps:
        step(PSO_setup['bounds'],
             PSO_setup['WriteFile'],
             PSO_setup['ReadFile'])

if __name__ == '__main__':
    main()
