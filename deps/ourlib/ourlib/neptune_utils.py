import argparse
import os

import socket
from munch import Munch
from path import Path
import tensorflow.keras as keras


def is_neptune_online():
  # I wouldn't be suprised if this would depend on neptune version
  return 'NEPTUNE_ONLINE_CONTEXT' in os.environ


_ctx = None


def _ensure_compund_type(input):
  if type(input) is not str:
    return input

  try:
    input = eval(input, {})
    if type(input) is tuple or type(input) is list:
      return input
    else:
      return input
  except:
    return input

def get_configuration(print_diagnostics=False, dict_prefixes=()):
  global _ctx
  if is_neptune_online():
    from deepsense import neptune
    _ctx = neptune.Context()
    exp_dir_path = os.getcwd()
    params = {k: _ensure_compund_type(_ctx.params[k]) for k in _ctx.params}
    _ctx.properties['pwd'] = os.getcwd()
    _ctx.properties['host'] = socket.gethostname()
  else:
    # local run
    parser = argparse.ArgumentParser(description='Debug run.')
    parser.add_argument('--ex', type=str)
    parser.add_argument("--exp_dir_path", default='/tmp')
    commandline_args = parser.parse_args()
    exp_dir_path = commandline_args.exp_dir_path
    params = {}
    if commandline_args.ex != None:
      vars = {'script': str(Path(commandline_args.ex).name)}
      exec(open(commandline_args.ex).read(), vars)
      spec_func = vars['spec']
      experiment = spec_func()[0] #take just the first experiment for testing
      params = experiment.parameters

  for dict_prefix in dict_prefixes:
    dict_params = Munch()
    l = len(dict_prefix)
    for k in list(params.keys()):
      if dict_prefix in k:
        dict_params[k[l:]] = params.pop(k)

    params[dict_prefix[:-1]] = dict_params

  if print_diagnostics:
    print("PYTHONPATH:{}".format(os.environ['PYTHONPATH']))
    print("cd {}".format(os.getcwd()))
    print(socket.gethostname())
    print("Params:{}".format(params))

  return Munch(params), exp_dir_path


def neptune_logger(m, v):
  global _ctx

  if _ctx is None or _ctx.experiment_id is None:
    print("{}:{}".format(m, v))
  else:
    _ctx.channel_send(name=m, x=None, y=v)


class NeptuneCallback(keras.callbacks.Callback):
  def __init__(self, report_every_batch=None, start_from_batch=0):
    self.report_every_batch = report_every_batch
    self.start_from_batch = start_from_batch
    self.epoch = 0

  def on_batch_end(self, batch, logs):
    if (self.report_every_batch and
            (batch >= self.start_from_batch or self.epoch > 0) and
            (batch % self.report_every_batch == 0)):
      for k in logs:
        if k not in ['batch', 'size', 'loss']:
          neptune_logger(k, logs[k])

  def on_epoch_end(self, epoch, logs):
    self.epoch = epoch
    prefix = "epoch_" if self.report_every_batch else ""
    for k in logs:
      if k not in ['val_loss', 'loss']:
        neptune_logger(prefix+k, logs[k])

  def on_epoch_begin(self, epoch, logs):
    self.epoch = epoch
    prefix = "epoch_" if self.report_every_batch else ""
    for k in logs:
      if k not in ['val_loss', 'loss']:
        neptune_logger(prefix+k, logs[k])


