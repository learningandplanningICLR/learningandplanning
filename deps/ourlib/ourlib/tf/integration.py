import time

from influxdb import InfluxDBClient

from ourlib.influxdb.client import create_influxdb_client, DEFAULT_INFLUXDB_HOST, DEFAULT_INFLUXDB_PORT, \
    DEFAULT_INFLUXDB_USERNAME, DEFAULT_INFLUXDB_PASSWORD, DEFAULT_INFLUXDB_DBNAME


class BaseTensorflowIntegrator(object):
    def add_summary(self, summary_writer, summary, global_step=None):
        raise NotImplementedError


class DummyTensorflowIntegrator(BaseTensorflowIntegrator):
    def add_summary(self, summary_writer, summary, global_step=None):

        from tensorflow.core.framework import summary_pb2  # pylint:disable=import-error,no-name-in-module

        if isinstance(summary, bytes):
            summ = summary_pb2.Summary()
            summ.ParseFromString(summary)
            summary = summ

        x = self._calculate_x_value(global_step)

        for value in summary.value:

            field = value.WhichOneof('value')

            if field == 'simple_value':
                self._send_numeric_value(summary_writer, value.tag, x, value.simple_value)

    def _send_numeric_value(self, summary_writer, value_tag, x, simple_value):
        pass
        # print('Captured value x = {}, y = {}'.format(x, simple_value))

    @staticmethod
    def _calculate_x_value(global_step):
        if global_step is not None:
            return global_step
        else:
            return time.time()


class InfluxDBTensorflowIntegrator(BaseTensorflowIntegrator):
    def __init__(self, experiment_id, client: InfluxDBClient):
        self.client = client
        self.experiment_id = experiment_id


    def add_summary(self, summary_writer, summary, global_step=None):
        from tensorflow.core.framework import summary_pb2  # pylint:disable=import-error,no-name-in-module

        if isinstance(summary, bytes):
            summ = summary_pb2.Summary()
            summ.ParseFromString(summary)
            summary = summ

        x = self._calculate_x_value(global_step)

        for value in summary.value:

            field = value.WhichOneof('value')

            if field == 'simple_value':
                self._send_numeric_value(summary_writer, value.tag, x, value.simple_value)

    def _send_numeric_value(self, summary_writer, value_tag, x, simple_value):
        # print('Captured value x = {}, y = {}'.format(x, simple_value))

        point = {
            "measurement": 'summaries',
            "time": x,
            "fields": {
                "value": simple_value,
                "time": time.time()

            },
            'tags': {
                'value_tag': value_tag,
                'experiment_id': self.experiment_id
            }
        }
        # print('Will send to influxdb {}'.format(point))
        self.client.write_points([point])

    @staticmethod
    def _calculate_x_value(global_step):
        if global_step is not None:
            return global_step
        else:
            return time.time()



def integrate_with_influxdb(experiment_id, influxdb_client=None,
                            host=DEFAULT_INFLUXDB_HOST,
                            port=DEFAULT_INFLUXDB_PORT,
                            user=DEFAULT_INFLUXDB_USERNAME,
                            password=DEFAULT_INFLUXDB_PASSWORD,
                            dbname=DEFAULT_INFLUXDB_DBNAME,
                            ):
    influxdb_client = influxdb_client or create_influxdb_client(host=host,
                                                                port=port,
                                                                user=user,
                                                                password=password,
                                                                dbname=dbname)
    integrator = InfluxDBTensorflowIntegrator(experiment_id,
                                              client=influxdb_client)

    awarelib_integrate_with_tensorflow(integrator)



def _awarelib_create_neptune_add_summary_wrapper(tensorflow_integrator, _FileWriter_add_summary_method):

    def _neptune_add_summary(summary_writer, summary, global_step=None, *args, **kwargs):
        tensorflow_integrator.add_summary(summary_writer, summary, global_step)
        _FileWriter_add_summary_method(summary_writer, summary, global_step, *args, **kwargs)

    return _neptune_add_summary


import tensorflow

def awarelib_integrate_with_tensorflow(tensorflow_integrator: BaseTensorflowIntegrator):

    _FileWriter_add_summary_method = tensorflow.summary.FileWriter.add_summary
    _FileWriter_add_graph_def_method = tensorflow.summary.FileWriter._add_graph_def

    tensorflow.summary.FileWriter.add_summary = \
        _awarelib_create_neptune_add_summary_wrapper(tensorflow_integrator, _FileWriter_add_summary_method)

    return tensorflow_integrator



