# -*- coding: utf-8 -*-
import six


class Experiment(object):

    def __init__(self, name, parameters, **kwargs):
        def _get_arg(k, sep=' '):
            list_type = ['tags', 'paths-to-copy', 'exclude', 'properties', 'python_path']
            v = kwargs.pop(k, [] if k in list_type else '')
            return v.split(sep) if isinstance(v, six.string_types) and k in list_type else v

        self.name = name[:16]
        self.parameters = parameters

        self.env = kwargs.pop('env', {})
        self.env['PYTHONPATH'] = ':'.join(['$PYTHONPATH', ] + _get_arg('python_path', sep=':'))

        for k in list(kwargs.keys()):
            self.__setattr__(k, _get_arg(k))

    def to_dict(self):
        return self.__dict__

