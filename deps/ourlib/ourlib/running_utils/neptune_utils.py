import os
import pprint

import yaml


from ourlib.tf.neptune_tf_integration import neptune_integrate_with_tensorflow, NeptuneAwarelibTensorflowIntegrator


def is_neptune_online():
    # I wouldn't be suprised if this would depend on neptune version
    return 'NEPTUNE_ONLINE_CONTEXT' in os.environ


def get_configuration():
    from deepsense import neptune
    if is_neptune_online():

        ctx = neptune.Context()
        exp_dir_path = os.environ.get('EXP_DIR_PATH', os.getcwd())
    else:
        # INFO: this is set by the local backend in mrunner, THIS IS A HACK!
        neptune_yaml_path = os.environ['NEPTUNE_YAML_PATH']
        exp_dir_path = os.environ.get('EXP_DIR_PATH', os.getcwd())

        with open(neptune_yaml_path, 'r') as stream:
            try:
                d = yaml.load(stream)
                pprint.pprint(d)
            except yaml.YAMLError as exc:
                print(exc)

        ctx = neptune.Context(offline_parameters=d['parameters'])

    return ctx, exp_dir_path

def neptune_integrate_with_tensorflow_hacked(neptune_ctx):
    if is_neptune_online():
        neptune_integrate_with_tensorflow(
            neptune_ctx,
            NeptuneAwarelibTensorflowIntegrator(neptune_ctx.job, neptune_ctx.job._api_service)
        )
