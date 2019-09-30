from distutils.version import LooseVersion

from neptune.internal.client_library.third_party_integration import _create_neptune_add_summary_wrapper, \
    _create_neptune_add_graph_def_wrapper, TensorflowIntegrator
from neptune.internal.common import NeptuneException


def neptune_integrate_with_tensorflow(job, tensorflow_integrator):

    try:
        import tensorflow
    except ImportError:
        raise NeptuneException('Requested integration with tensorflow while '
                               'tensorflow is not installed.')

    # pylint: disable=no-member, protected-access, no-name-in-module, import-error


    version = LooseVersion(tensorflow.__version__)

    if LooseVersion('0.11.0') <= version < LooseVersion('0.12.0'):

        _add_summary_method = tensorflow.train.SummaryWriter.add_summary
        _add_graph_def_method = tensorflow.train.SummaryWriter._add_graph_def

        tensorflow.train.SummaryWriter.add_summary = \
            _create_neptune_add_summary_wrapper(tensorflow_integrator, _add_summary_method)
        tensorflow.train.SummaryWriter._add_graph_def = \
            _create_neptune_add_graph_def_wrapper(tensorflow_integrator, _add_graph_def_method)

    elif (LooseVersion('0.12.0') <= version < LooseVersion('0.13.0')) or (
            LooseVersion('1.0.0') <= version):

        _add_summary_method = tensorflow.summary.FileWriter.add_summary
        _add_graph_def_method = tensorflow.summary.FileWriter._add_graph_def

        tensorflow.summary.FileWriter.add_summary = \
            _create_neptune_add_summary_wrapper(tensorflow_integrator, _add_summary_method)
        tensorflow.summary.FileWriter._add_graph_def = \
            _create_neptune_add_graph_def_wrapper(tensorflow_integrator, _add_graph_def_method)

    else:
        raise NeptuneException("Tensorflow version {} is not supported.".format(version))

    return tensorflow_integrator


class NeptuneAwarelibTensorflowIntegrator(TensorflowIntegrator):
    # INFO: this is a hack, we want to change the name of the channel!
    def _get_channel(self, summary_writer, value_tag, channel_type):
        # pylint: disable=protected-access
        # writer_name = self.get_writer_name(summary_writer.get_logdir())
        channel_name = '{}'.format(value_tag)
        return self._job.create_channel(channel_name, channel_type)

