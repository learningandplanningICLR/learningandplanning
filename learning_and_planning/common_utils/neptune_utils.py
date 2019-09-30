import argparse
import os


from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

import numpy as np


def is_neptune_online():
  # I wouldn't be suprised if this would depend on neptune version
  return 'NEPTUNE_ONLINE_CONTEXT' in os.environ


def get_configuration():
  from deepsense import neptune
  if is_neptune_online():
    # running under neptune
    ctx = neptune.Context()
    # I can't find storage path in Neptune 2 context
    # exp_dir_path = ctx.storage_url - this was used in neptune 1.6
    exp_dir_path = os.getcwd()
  else:
    # local run
    parser = argparse.ArgumentParser(description='Debug run.')
    parser.add_argument('--ex', type=str)
    parser.add_argument("--exp_dir_path", default='/tmp')
    commandline_args = parser.parse_args()
    if commandline_args.ex != None:
      vars = {}
      exec(open(commandline_args.ex).read(), vars)
      spec_func = vars['spec']
      # take first experiment (params configuration)
      experiment = spec_func()[0]
      params = experiment.parameters
    else:
      params = {}
    # create offline context
    ctx = neptune.Context(offline_parameters=params)
    exp_dir_path = commandline_args.exp_dir_path
  return ctx, exp_dir_path


def render_figure(figure, size=None):
  if size is not None:
    (height, width) = size
    dpi = 100
    figure.set_dpi(dpi)
    figure.set_size_inches(height / dpi, width / dpi)
  canvas = FigureCanvasAgg(figure)
  canvas.draw()
  (buf, (width, height)) = canvas.print_to_buffer()
  image_rgba = np.fromstring(buf, dtype=np.uint8).reshape((height, width, 4))
  return image_rgba[:, :, :3]


def send_image(channel_name, x, image, description='', neptune_ctx=None):
  from deepsense import neptune
  image_pil = Image.fromarray(image)

  image_neptune = neptune.Image(
      name=str(x),
      data=image_pil,
      description=description,
  )
  if neptune_ctx is None:
    (neptune_ctx, _) = get_configuration()
  neptune_ctx.channel_send(channel_name, x=x, y=image_neptune)


def send_figure(channel_name, x, figure, description='', neptune_ctx=None):
  image_np = render_figure(figure)
  send_image(channel_name, x, image_np, description, neptune_ctx)
