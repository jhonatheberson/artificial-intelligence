"""GROMACS helpers.

This module contains classes and helper functions to edit topology files,
create directory structures to run GROMACS simulations, make the GROMACS
command-line calls such as ``grompp`` and ``mdrun``.

"""
import subprocess

from pathlib import Path

import numpy as np
import panedr


class PES():
    """
    Class to run a relaxed PES scan in GROMACS.

    Attributes
    ----------
    reference : array_like
        Reference data array, taken as the fitting goal. If `reference` is
        two-dimensional, the first column should contain the PES
        coordinates and the corresponding potential energy values should be
        in the second column, and `points` must be `None`.  If `reference`
        is one-dimensional, `points` must be provided and must have the
        same number of elements as `reference`.
    points : array_like
        PES coordinates corresponding to the potential energy values
        present in `reference`. Must not be provided if `reference` is
        two-dimensional.
    basedir : str or path_like, optional
        Directory in which the gromacs files will be written. Must not
        exist (the default is ``./run``).
    topfile : str or path_like, optional
        Gromacs ``top``-file with the ``#define`` statements to be updated
        with the PSO particle coordinates (the default is
        ``top/torsion.top``).
    mdpfile : str or path_like, optional
        Gromacs ``mdp``-file with the simulation parameters (the default is
        ``mdp/em.mdp``).
    basename : str, optional
        Filename root for the ``.gro`` files (the default is ``torsion``).
    zero : bool, optional
        Shift the PES such that the minimum is at zero (default: `True`).

    """

    def __init__(self, reference, points=None, basedir=None, topfile=None,
                 mdpfile=None, grodir=None, basename="torsion", zero=True,
                 gmxcmd='gmx'):
        self.reference = np.array(reference)
        if basedir is None:
            Path.cwd().joinpath('run')
        else:
            self.basedir = Path(basedir).resolve()
        self.basedir.mkdir(parents=True)
        self.basename = basename
        self.zero = zero
        self.gmxcmd = gmxcmd
        # TODO Clean up these presets.
        # TODO Check if base files exist.
        if topfile is None:
            self.topfile = self.basedir.joinpath('..', 'top',
                                                 f'{self.basename}.top')
        else:
            self.topfile = Path(topfile).resolve()
        if mdpfile is None:
            self.mdpfile = self.basedir.joinpath('..', 'mdp', 'em.mdp')
        else:
            self.mdpfile = Path(mdpfile).resolve()
        if grodir is None:
            self.grodir = self.basedir.joinpath('..', 'gro')
        else:
            self.grodir = Path(grodir).resolve()
        if self.reference.ndim == 1:
            if points is None:
                raise(ValueError, "Reference has only one dimension."
                      + "Points needed.")
            elif len(points) != self.reference.shape[0]:
                # TODO Issue a warning in the logfile reporting default
                # behavior.
                raise(ValueError, f"Reference has {self.reference.shape[0]} "
                      + f"entries while points has {len(points)} entries. "
                      + "They must be equal.")
            self.points = np.array(points)
        else:
            if points is not None:
                raise(ValueError, "Reference has two dimensions and points"
                      + " were given. This is unsupported.")
            self.points = self.reference[:, 0]
        if self.zero:
            if reference.ndim == 1:
                self.reference -= np.min(self.reference)
            else:
                self.reference[:, 1] -= np.min(self.reference[:, 1])

    def edit_top(self, x, point, destdir):
        """
        Make a topology file for the current PSO coordinates.

        Parameters
        ----------
        x : array_like
            PSO particle coordinates.
        point : int or float
            PES coordinate for the current run.
        destdir : path_like
            Directory to write the new topology.

        Returns
        -------
        desttop : path_like
            Name of the generated topology file.
        """
        desttop = destdir.joinpath(f'topol_{point:04d}.top')
        newtop = []
        with open(self.topfile, 'r') as top:
            for line in top:
                if 'PSO FIT GOES HERE' in line:
                    newtop.append(f'#define ANG {point:04f}\n')
                    for i, x_i in enumerate(x):
                        newtop.append(f'#define FIT_{i:02d} {x_i}\n')
                else:
                    newtop.append(line)
        with open(desttop, 'w') as dest:
            dest.writelines(newtop)
        return desttop

    def prepare(self, x, idx, step_num):
        """
        Create directories and tor files to run one PES scan.

        Parameters
        ----------
        x : array_like
            PSO particle coordinates.
        idx : int or None
            PSO particle index.
        step_num : int or None
            PSO run step number.

        Returns
        -------
        stepdir : path_like
            Directory with the ``tpr`` and related files to be passed to
            ``gmx mdrun``.
        """

        if idx is None and step_num is None:
            i = 0
            while self.basedir.joinpath(f'step_{i+1:05d}/run/').is_dir():
                # step_00000/run two levels because that's what our .itp files expect.
                i += 1
            else:
                stepdir = self.basedir.joinpath(f'step_{i+1:05d}/run/')
        elif idx is not None and step_num is not None:
            stepdir = self.basedir.joinpath(f'step_{step_num:03d}',
                                            f'part_{idx:04d}')
        else:
            raise ValueError("Both part_idx and step_num must be integers or None. Only one set as None is not supported.")

        stepdir.mkdir(parents=True)

        for point in self.points:
            grompp = subprocess.run([self.gmxcmd, 'grompp',
                                     '-f', self.mdpfile,
                                     '-p', self.edit_top(x, point, stepdir),
                                     '-c', self.grodir.joinpath(f'{self.basename}_{point:04d}.gro'),
                                     '-o', stepdir.joinpath(f'{self.basename}_{point:04d}')],
                                    capture_output=True)
            if grompp.returncode != 0:
                raise RuntimeError(f'grompp process called as {grompp.args}'
                                   + 'failed:\n' + grompp.stderr.decode())
        return stepdir

    def runmd(self, stepdir):
        """
        Call ``gmx mdrun``.

        Parameters
        ----------
        path_like
            Directory with the ``tpr`` file.

        """
        for point in self.points:
            gmx = subprocess.run([self.gmxcmd, 'mdrun', '-ntomp', '1',
                                  '-deffnm',
                                  stepdir.joinpath(f'{self.basename}_{point:04d}')],
                                 capture_output=True)
            if gmx.returncode != 0:
                raise RuntimeError(f"mdrun process called as {gmx.args}"
                                   + "failed:\n" + gmx.stderr.decode())
            # Delete the tpr and trr files as they take too much disk space.
            # TODO: make optional
            stepdir.joinpath(f'{self.basename}_{point:04d}.tpr').unlink()
            stepdir.joinpath(f'{self.basename}_{point:04d}.trr').unlink()

    def analyze(self, stepdir):
        """
        Calculate the sumsquare of the PES differences.

        Parameters
        ----------
        stepdir : path_like
            Directory with the ``edr`` files after a successful ``mdrun``.

        Returns
        -------
        float
            Fitness value.

        """
        torsioncurve = np.zeros(len(self.points))
        for i, point in enumerate(self.points):
            energies = panedr.edr_to_df(
                    stepdir.joinpath(f'{self.basename}_{point:04d}.edr'))
            torsioncurve[i] = energies[u'Potential'].tail(1)
        if self.zero:
            torsioncurve -= np.min(torsioncurve)
        diff = torsioncurve - self.reference
        return np.sum(diff * diff)

    def fitness(self, x, part_idx=None, step_num=None):
        """
        Perform a full PES calculation and return the sum square of the
        differences with the reference.

        Parameters
        ----------
        x : array_like
            PES particle coordinates.
        part_idx : int
            PES particle index.
        step_num=0 : int
            PES run step number.

        Returns
        -------
        float
            Fitness value, calculated as the sum of square the differences
            between the reference PES and the obtained one.

        """
        stepdir = self.prepare(x, part_idx, step_num)
        self.runmd(stepdir)
        return self.analyze(stepdir)
