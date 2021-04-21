"""Manages loading and dispatching jobs for fitness testing."""

import importlib.util


def load_function(file_path, function_name):
    """Run function_name at file file_path with kwargs arguments."""
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    # https://gitlab.com/obestwalter/pico-pytest
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # TODO: manage errors/exceptions...
    return getattr(module, function_name)


def create_test(setup_script, run_script, eval_script):
    """Create a container dictionary for the setup/run/eval funcs."""
    fitness_test = {}

    fitness_test['setup'] = load_function(setup_script, 'setup')
    fitness_test['run'] = load_function(run_script, 'run')
    fitness_test['evaluate'] = load_function(evaluate_script, 'evaluate')

    return fitness_test

def load_hooks(obj, hookfiles):
    """Load hooks from configparse dict into callable functions."""
    #  TODO check validity of hook scripts
    # TODO Methods to call hooks?
    obj.load_hooks = load_hooks
    # TODO Test line above for bugs.
    obj.hookfiles = hookfiles
    obj.hooks = {}
    for hookname, hookfile in hookfiles:
        obj.hooks[hookname] = load_function(hookfile, hookname)

def load_tests(obj, testfiles):
    """Load fitness tests from configparse dict into callable functions."""
    # TODO check validity of setup, run and eval scripts
    obj.load_tests = load_tests
    # TODO Test line above for bugs.
    obj.testfiles = testfiles
    obj.tests = {}
    for testname, test in testfiles.items():
        obj.tests[testname] = create_test(test['setup'],
                                          test['run'],
                                          test['eval'])
        obj.tests[testname]['weight'] = test['weight']
    # NOTE: Depends on python 3.6 ordered dictionary.
    obj.test_weights = np.array([t['weight'] for t in obj.tests])

def batch_exec(fitness_test, population,
               *args,
               jobs_per_batch=None,
               **kwargs):
    """Launch batch jobs with given batch size."""
    # TODO: debug level logging
    test_output = []
    if jobs_per_batch is None or jobs_per_batch == 0:
        jobs_per_batch == len(population)
    for i in range(0, len(population), jobs_per_batch):
        test_output.append(fitness_test(population[i:i+jobs_per_batch],
                                        *args, **kwargs))
    return test_output
