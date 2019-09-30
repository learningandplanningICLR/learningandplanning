import sys
from collections import defaultdict
import numpy as np

import tensorflow as tf

from ourlib.image.utils import convert_to_PIL
from ourlib.logger import logger
import io


def create_simple_summary(tag, y):
    return tf.Summary(value=[tf.Summary.Value(tag=tag,
                                              simple_value=y)])


def create_image_summary(tag, image):
    image_pil = convert_to_PIL(image)

    with io.BytesIO() as output:
        image_pil.save(output, format="PNG")

        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=output.getvalue(),
                                   height=image_pil.size[0],
                                   width=image_pil.size[1])
        # Create a Summary value
        res = tf.Summary.Value(tag=tag,
                               image=img_sum)
    return tf.Summary(value=[res])


def add_simple_summary(tag, y, summary_writer, global_step=None):
    summary = create_simple_summary(tag, y)
    summary_writer.add_summary(summary, global_step=global_step)


def add_image_summary(tag, image, summary_writer, global_step=None):
    summary = create_image_summary(tag, image)
    summary_writer.add_summary(summary, global_step=global_step)


def mean_aggregator(l):
    return np.mean(l)


class SummaryHelper(object):
    def __init__(self, summary_writer):
        self.summary_writer = summary_writer
        self.channels = defaultdict(list)

    def add_simple_summary(self, tag, y, freq=20, global_step=None, aggregator=mean_aggregator,
                           force_neptune=False, neptune_ctx=None):
        self.channels[tag].append(y)
        if len(self.channels[tag]) >= freq:
            aggregated_y = aggregator(self.channels[tag])

            logger.debug(
                'Will send value, {}, {}, {}, force_neptune = {}'.format(tag, global_step, aggregated_y, force_neptune))
            add_simple_summary(tag=tag,
                               global_step=global_step,
                               y=aggregated_y,
                               summary_writer=self.summary_writer)
            if force_neptune:
                assert neptune_ctx is not None
                neptune_ctx.channel_send(tag,
                                         x=global_step,
                                         y=aggregated_y)
            # self.summary_writer.flush()
            self.channels[tag] = []

    def add_image_summary(self, tag, img, global_step=None,
                           force_neptune=False, neptune_ctx=None):
        add_image_summary(tag=tag,
                          global_step=global_step,
                          image=img,
                          summary_writer=self.summary_writer)
        if force_neptune:
          assert neptune_ctx is not None
          neptune_ctx.channel_send(tag,
                                   x=global_step,
                                   y=img)
