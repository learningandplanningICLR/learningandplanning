import datetime
import os
import tempfile

import os.path as osp
import baselines
from baselines.logger import make_output_format, KVWriter, Logger, log

LOG_OUTPUT_FORMATS     = ['stdout', 'log', 'csv']

def configure(dir, format_strs=None, custom_output_formats=None):
    if not dir:
        return 

    assert isinstance(dir, str)
    os.makedirs(dir, exist_ok=True)

    if format_strs is None:
        strs = os.getenv('OPENAI_LOG_FORMAT')
        format_strs = strs.split(',') if strs else LOG_OUTPUT_FORMATS
    output_formats = [make_output_format(f, dir) for f in format_strs]

    if custom_output_formats is not None:
        assert isinstance(custom_output_formats, list)
        for custom_output_format in custom_output_formats:
            assert isinstance(custom_output_format, KVWriter)
        output_formats.extend(custom_output_formats)

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats)
    log('Logging to %s' % dir)


# Monkey-patches PR to openai-baselines. See PR at https://github.com/openai/baselines/pull/363
def monkey_patch():
    baselines.logger.configure = configure