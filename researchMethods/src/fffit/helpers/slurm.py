"""Manage job submission via SLURM.

Heavly based on https://pypi.org/project/clusterlib/
"""

from collections import namedtuple
import subprocess


def submit(job_script, *, job_name="job_by_dispatcher", time="24:00:00",
           partition="cluster", nodes=1, cpus_per_task=None, array=None,
           afterany=None, shell='#!/bin/bash', **sbatch_opts):
    """Call sbatch with given options and job_script as a script piped to it.

    Parameters
    ----------
    job_script : str,
        Script to be submitted. Should be the contents of the `job.sh` file if
        the user submitted by calling `sbatch job.sh`.

    job_name :  str, optional (default="job_by_dispatcher")
        Name of the job, as in sbatch --job-name option.

    time : str, optional (default="24:00:00")
        Time limit, as in sbatch `--time` option.

    partition : str, optional (default="cluster")
        Partition (queue) name, as expected in sbatch `--partition` option.

    nodes : str or int, optional (default="1")
        Number of nodes specified as in sbatch `--nodes`.

    cpus_per_task : int, optional (default=None)
        Number of cpus per task specified as in sbatch `--cpus-per-task`.

    array : str, optional (default=None)
        Array settings expressed as a sequence or range on indexes, as per
        sbatch `--array` option.

    afterany : str or list, optional (default=None)
        List of dependencies for this job, as specified in sbatch
        `--dependency=afterany`.

    **sbatch_opts : dict (as list of keyword arguments)
        Extra options passed to sbatch, such that key=val is added to the
        arguments as `--key=val`.

    Returns
    -------
    returncode, int
        Return code givem by sbatch. 0 = success.

    jobid, int or None
        Job ID as interpreted from the standard output.

    stdout, str
        sbatch's stdandard output.

    stderr, str
        sbatch's standard error. If returncode == 0, this should be empty.

    """
    args = ['sbatch',
            '--job-name=' + job_name,
            '--time=' + time,
            '--partition=' + partition,
            '--nodes=' + str(nodes)]

    if cpus_per_task is not None:
        args.append('--cpus-per-task=' + str(cpus_per_task))
    if array is not None:
        if isinstance(array, str):
            arraystr = array
        elif isinstance(array, list):
            arraystr = ','.join(map(str, array))
            # TODO: detect range formats?
            # For example: if array == list(range())
        else:
            raise TypeError(f"Invalid format for array: {array}")
        args.append('--array=' + arraystr)
    if afterany is not None:
        # TODO: implement multiple dependencies.
        dependencystr = 'afterany:'
        if isinstance(afterany, str):
            dependencystr += afterany
        elif isinstance(afterany, list):
            dependencystr += ':'.join(map(str, afterany))
        else:
            raise TypeError(f"Invalid format for afterany: {afterany}")
        args.append('--dependency=' + dependencystr)
    for option, value in sbatch_opts.items():
        args.append('--' + option + '=' + str(value))

    if shell is not None:
        if not job_script.startswith('#!'):
            job_script = '\n'.join((shell, job_script))

    # TODO: Print args if loglevel=DEBUG
    print(' '.join(args))
    sbatch = subprocess.run(args, input=job_script,
                            stderr=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            universal_newlines=True)
    if len(sbatch.stdout) > 0:
        jobid = int(sbatch.stdout.split()[-1])
    else:
        jobid = None
    result = namedtuple("result", ["returncode", "jobid", "stdout", "stderr"])
    # https://docs.quantifiedcode.com/python-anti-patterns/readability/
    return result(sbatch.returncode, jobid, sbatch.stdout, sbatch.stderr)


def submit_script(jobfile, *args, **kwargs):
    """Call the submit function with the contents of `jobfile'.

    Parameters
    ----------
    jobfile :  str, Path
        File containing the job script to be submitted.

    *args, **kwargs: optional
        Arguments to submit().

    Returns
    -------
    returncode, int
        Return code givem by sbatch. 0 = success.

    jobid, int or None
        Job ID as interpreted from the standard output.

    stdout, str
        sbatch's stdandard output.

    stderr, str
        sbatch's standard error. If returncode == 0, this should be empty.
    """
    with open(jobfile, 'r') as f:
        job_script = f.read()

    return submit(job_script, *args, **kwargs)
