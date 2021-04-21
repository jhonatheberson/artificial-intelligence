"""Algorithm optimization of particle swarm (PSO).

The particle optimization algorithm is a metaheuristic, it attempts to optimize
a problem interactively with a swarm of particles percoorendo to a mathematical
function or search space.

the algorithm has the following steps:
    1 - Creates a sample space, which is the swarm of particles where it is
        demilitarized by the mathematical function.
    2 - then it updates all the particles with their positions and volecidades
        thus sweeping the function and obtaining the best result of this
        function.

"""
import multiprocessing as mp
import sys

import numpy as np

# TODO: inherit logger
#logging.basicConfig(filename='output.log', level=logging.DEBUG,
#                    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')


class Particle(object):
    """Creates the particle and updates position and velocity."""

    def __init__(self, x0, bounds, w=0.5, c=(2,2), sigma=None, vsigma=None):
        """Initialize the particle.

        Args:
            :param x0(str): Initial value for the sample space to create the
                            Gaussian.
            :param bounds(:obj:`list` of :obj:`str`): Limits for the sample
                            space to create the Gaussian.
        """
        self.pos_best = []  # Best position individual.
        self.fitness_best = None  # Error best individual.
        self.curr_fitness = None
        self.w = w
        self.c = c
        bounds = np.array(bounds)
        if sigma is None:
            sigma = np.abs(bounds[1] - bounds[0])
        elif isinstance(sigma, float) or isinstance(sigma, int):
            sigma = np.abs(bounds[1] - bounds[0])/sigma
        self.position = np.random.normal(x0, sigma)
        if vsigma is None:
            vsigma = np.abs(bounds[1] - bounds[0])
        elif isinstance(vsigma, float) or isinstance(vsigma, int):
            vsigma = np.abs(bounds[1] - bounds[0])/vsigma
        self.velocity = np.random.normal(np.zeros(len(x0)), vsigma)

    def check_fitness(self):
        """Update personal best fitness."""
        # Check to see if the current position is an individual best:
        if self.fitness_best is None or self.curr_fitness < self.fitness_best:
            self.pos_best = self.position
            self.fitness_best = self.curr_fitness

    def update_velocity(self, pos_best_g):
        """Update new particle velocity.

        Args:
            :param pos_best_g(str): best overall swarm position.

        Returns:
            :return: Void.

        """
        # TODO Make these adjustable parameters
        r1 = np.random.random(len(self.velocity))
        r2 = np.random.random(len(self.velocity))
        vel_cognitive = self.c[0] * r1 * (self.pos_best - self.position)
        vel_social = self.c[1] * r2 * (pos_best_g - self.position)
        self.velocity = self.w * self.velocity + vel_cognitive + vel_social

    def update_position(self, bounds):
        """Update the particle position based off new velocity updates.

        Args:
            :param bounds(:obj:`list` of :obj:`str`): Limits for the sample
            space to create the Gaussian.

        Returns:
            :return: Void.
        """
        self.position += self.velocity

        # TODO Deal with velocities when particle goes out of bounds
        np.clip(self.position, bounds[0], bounds[1], out=self.position)
        np.clip(self.velocity, bounds[0], bounds[1], out=self.velocity)
        self.velocity[np.isclose(self.position, bounds[0])] *= -1
        self.velocity[np.isclose(self.position, bounds[1])] *= -1


class PSO(object):
    """Contains the population and methods for performing steps."""

    def __getstate__(self):
        """Remove unpickable entries from object.

        Currently, removes fitness tests as callable functions.
        """
        state = self.__dict__.copy()
        del state['tests']
        if 'hooks' in state:
            del state['hooks']
        return state

    def __setstate__(self, state):
        """Recover unpickable items to restore original object.

        Currently, calls self.load_tests in order to get callable fitness
        tests and self.load_hooks to get pre_ and _post step hooks.
        """
        self.__dict__.update(state)
        if 'testfiles' in self.__dict__:
            # TODO: log
            self.load_tests()
        if 'hookfiles' in self.__dict__:
            # TODO: log
            self.load_hooks(self.hookfiles)

    def __init__(self, maxiter=None, goal=1.0, w=0.5, c = (2,2), submit_to_cluster=False):
        """Initialize the PSO object."""
        self.ncpu = 1
        self.goal = goal
        self.w = w
        self.c = c
        self.submit_to_cluster = submit_to_cluster
        self.fitness = None
        self.step_number = 0
        self.maxiter = maxiter
        self.swarm = None
        if self.submit_to_cluster:
            # TODO: correctly handle the cluster and multitest cases.
            raise NotImplementedError('Cluster submission in review.')

    def populate(self, num_particles, x0=None, bounds=None, sigma=None,
                 vsigma=None):
        """Create the population of particles that is the swarm.

        Args:
            :param num_particles(:obj:`int`): Number of particles to be
                created.
            :param initial(): Initial value for the sample space to create the
                Gaussian.
            :param bounds(:obj:`list` of :obj:`str`): Limits for the sample
                space to create the Gaussian.

        Returns:
            :return swarm(:obj:`list` of :obj:`Particles`): a list of swarms.
        """
        if self.swarm is None:
            self.bounds = bounds
            self.swarm = [Particle(x0, bounds, w=self.w, c=self.c,
                          sigma=sigma, vsigma=vsigma) for i in range(num_particles)]
        else:
            raise RuntimeError("Tried to populate non-empty swarm")

    def evaluate_single_fitness_test(self, func,
                                     enum_particles=False, add_step_num=False,
                                     **kwargs):
        """Run the given function as the fitness test for all particles.

        Parameters:
        -----------
        fun : callable
            The fitness test function to be minimized:

                ``func(particle.position, **kwargs) -> float``.

        enum_particles : boolean
            If `True`, the swarm will be enumerated and the particle index will
            be passed to `func` as keyword `part_idx`, added to `kwargs`

        add_step_num : boolean
            If `True`, the current step number will be passed to `func`
            as keyword `step_num`, added to `kwargs`

        **kwargs: Other keywords to the fitness function, will be passed as is.
        """
        if add_step_num:
            kwargs['step_num'] = self.step_number
        if self.ncpu == 1:
            if enum_particles:
                for part_idx, particle in enumerate(self.swarm):
                    kwargs['part_idx'] = part_idx
                    particle.curr_fitness = func(particle.position, **kwargs)
            else:
                for particle in self.swarm:
                    particle.curr_fitness = func(particle.position, **kwargs)
        elif self.ncpu > 1:
            with mp.Pool(processes=self.ncpu) as pool:
                argslist = []
                p = []
                for part_idx, particle in enumerate(self.swarm):
                    argslist.append(dict(kwargs))
                    # argslist[-1]['x'] = particle.position
                    if enum_particles:
                        argslist[-1]['part_idx'] = part_idx
                for idx, args in enumerate(argslist):
                    p.append(pool.apply_async(func, args=(self.swarm[idx].position,),kwds=args))
                results = [ r.get() for r in p ]
            for part_idx, particle in enumerate(self.swarm):
                particle.curr_fitness = results[part_idx]

    def calculate_global_fitness(self):
        """Calculate the fitness of the function or sample space.

        Returns:
            :return fitness(:obj:`float`): Returns the fitness of the function
            or sample space.
        """
        self.swarm_radius = 0
        for particle in self.swarm:
            particle.check_fitness()
            # determine if current particle is the best(globally)
            if self.fitness is None or particle.curr_fitness < self.fitness:
                self.pos_best_glob = np.array(particle.position)
                self.fitness = float(particle.curr_fitness)
        # Stop criteria
        for particle in self.swarm:
            dist = np.linalg.norm(particle.position - self.pos_best_glob)
            if dist > self.swarm_radius:
                self.swarm_radius = dist
        return self.fitness     # Do we actually need to return something?

    def update_swarm(self):
        """Update the swarm with new positions and speeds.

        Returns:
            :return swarm(:obj:`list` of :obj:`Particles`): returns a list of
            swarms.
        """
        if self.fitness is None:
            logging.error("Cannot update the swarm before calculating Fitness")
            raise RuntimeError("Updated the swarm before calculating Fitness")
        # cycle through swarm and update velocities and position
        for particle in self.swarm:
            particle.update_velocity(self.pos_best_glob)
            particle.update_position(self.bounds)
        if self.submit_to_cluster:
            self.curr_iter['update'] += 1

    def do_full_step(self, func, **kwargs):
        """Perform a full PSO step.

        This method goes through all other methods in order to perform a full
        PSO step, so it can be called from a loop in the run() method.
        """
        if self.fitness is not None and self.step_number < self.maxiter:
            self.update_swarm()
        if self.submit_to_cluster:
            raise NotImplementedError('Multistep jobs are under revision.')
        else:
            self.evaluate_single_fitness_test(func, **kwargs)
        self.calculate_global_fitness()
        self.step_number += 1

    def run(self, func, PSO_DEBUG=None, **kwargs):
        """Perform a full optimization run.

        Does the optimization with the execution of the update of the speeds
        and coordinates also checks the criterion stopped to find fitnnes.

        Parameters
        ----------
        func : callable
            Function that calculates fitnnes.

        Returns
        -------
            The dictionary that stores the optimization results.
        """
        self.swarm_radius = None
        # TODO make a better radius-based stop criterion.
        while (self.swarm_radius is None or
               self.step_number < self.maxiter and
               self.swarm_radius > 1e-3):
            self.do_full_step(func, **kwargs)
            if PSO_DEBUG is not None:
                with open(PSO_DEBUG, 'a') as dbg_file:
                    curr_best = min([p.curr_fitness for p in self.swarm])
                    print(f"# {self.step_number} {curr_best} {self.fitness}")
                    print(f"\n\n# {self.step_number} {curr_best} {self.fitness}",
                          file=dbg_file)
                    np.savetxt(dbg_file,
                               [(*p.position, p.curr_fitness)
                                   for p in self.swarm])
            if self.fitness < self.goal:
                break
        self.results = {}
        self.results['best_pos'] = self.pos_best_glob
        self.results['fitness'] = self.fitness
        return self.results
