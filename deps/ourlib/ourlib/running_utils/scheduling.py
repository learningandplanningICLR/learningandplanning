import argparse
from typing import Optional

D = {}


def register_generator(alias, add_alias:Optional[bool]=False):

    def real_decorator(func):
        def wrapper(*args, **kwargs):
            speces = func(*args, **kwargs)
            if add_alias:
                for spec in speces:
                    spec['alias'] = alias
            return speces

        if alias:
            D[alias] = wrapper
        D[func.__name__] = wrapper
        if 'default' not in D:
            D['default'] = wrapper

        return wrapper

    return real_decorator


def get_registered_generators():
    return D


def get_generator_by_name(name):
    generators_dict = get_registered_generators()
    if name not in generators_dict:
        raise RuntimeError('Unknown configuration generator {}, available = {}'.format(
            name, list(generators_dict.keys())
        ))
    gen_func = D[name]
    return gen_func


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='default')
    return parser


def standard_spec(create_experiment_for_spec_fn, argv):

    parser = create_parser()
    args, rest_argv = parser.parse_known_args(argv)
    print(args)
    all_generators = get_registered_generators()
    gen_func = get_generator_by_name(args.name)
    print('All generators: {}'.format('\n'.join(all_generators.keys())))
    print('Params configurations generator = {}'.format(gen_func.__name__))
    params_configurations = gen_func()

    experiments = [create_experiment_for_spec_fn(**params) for params in params_configurations]
    print(len(experiments))
    return experiments
