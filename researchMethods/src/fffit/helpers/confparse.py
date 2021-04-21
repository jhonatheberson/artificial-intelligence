"""Parses the configuration file in INI format."""

from pathlib import Path
import configparser


def read_conf(conffile):
    general_sections = ['PSO', 'HOOKS', 'DEFAULT']
    config_path = Path(conffile)
    if not config_path.exists():
        logging.error(f"{config_path} not found.")
        raise FileNotFoundError(f"Error: file {config_path} not found")

    config = configparser.ConfigParser()
    config.read(conffile)

    if 'PSO' not in config.keys():
        logging.error(f'PSO section missing in config file {conffile}')
        raise FileNotFoundError
        # TODO: Is this the correct exception?
    ReadFile = (config['PSO']['ReadFile'])

    WriteFile = config['PSO'].get('WriteFile', ReadFile)
    if (WriteFile == ''):
        WriteFile = ReadFile

    num_dimens = int(config['PSO']['num_dimens'])
    num_particles = int(config['PSO']['num_particles'])
    maxiter = int(config['PSO']['maxiter'])
    initial = np.fromstring(config['PSO']['initial'], dtype=float, sep=' ')
    upper_bounds = np.fromstring(config['PSO']['Upper_bounds'],
                                 dtype=float, sep=' ')
    lower_bounds = np.fromstring(config['PSO']['Lower_bounds'],
                                 dtype=float, sep=' ')
    if(len(upper_bounds) == len(upper_bounds) == num_dimens):
        bounds = [lower_bounds, upper_bounds]
    else:
        logging.debug("Number of particle dimensions not equal to bounds size")

    # Now read the hook scripts locations
    # Hook scripts will be run after all setup(), run() and evaluate() fitness
    # test scripts are called. They are intended primaly to submit a cluster
    # job to recall our main script which depends on the exit status of all the
    # fitness test jobs. In case of SLURM jobs, it's usually a command like
    # sbatch --afterok=xxxx:yyyy:zzzz
    if 'HOOKS' in config:
        available_hooks = [ 'pre_setup',
                            'post_setup',
                            'pre_run',
                            'post_run',
                            'pre_eval',
                            'post_eval' ]
        hooks = {}
        for hook in available_hooks:
            if hook in config['HOOKS']:
                hooks[hook] = Path(config['HOOKS'].get(hook))
                # TODO check validity of hook scripts

    # Now read the fitness test script locations
    fitness_tests = {}
    for k in config.keys():
        if k in general_sections:
            continue
        setup_script = Path(config[k]['setup'])
        run_script = Path(config[k]['run'])
        eval_script = Path(config[k]['eval'])
        # TODO check validity of setup, run and eval scripts
        fitness_tests[k] = {}
        fitness_tests[k]['setup'] == setup_script
        fitness_tests[k]['run'] == run_script
        fitness_tests[k]['eval'] == eval_script
        fitness_tests[k]['weight'] = float(config[k].get('weight', '1.0'))

    PSO_setup = {"ReadFile": ReadFile,
                 "WriteFile": WriteFile,
                 "num_dimens": num_dimens,
                 "num_particles": num_particles,
                 "maxiter": maxiter,
                 "initial": initial,
                 "bounds": bounds,
                 "fitness_tests": fitness_tests,
                 "hooks": hooks}
    return PSO_setup
