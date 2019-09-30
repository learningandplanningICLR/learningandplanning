from baselines.logger import Logger
from mrunner.helpers.client_helper import logger
import numpy as np


class SummaryBaselinesLogger(Logger):
    def __init__(self, *args, **kwargs):
        summary_helper = kwargs.pop('summary_helper')
        super().__init__(*args, **kwargs)

        self.summary_helper = summary_helper
        self.global_step = 0

    def dumpkvs(self):
        # print(10 * 'dumpkvs\n')
        # print(self.name2val.items())
        for k, v in self.name2val.items():
            self.summary_helper.add_simple_summary(tag=k, y=v, freq=1,
                                                   global_step=self.global_step
                                                   )
        self.global_step += 1

        # INFO: we call parent at the end because it will clear name2val dicts
        super().dumpkvs()

# TODO: This to be refactored and included to mrunner may be
from baselines.logger import Logger


class NeptuneLogger(Logger):

    def __init__(self, *args, **kwargs):
        self.dilution = kwargs.pop('dilution', 1)
        super().__init__(*args, **kwargs)
        self.counter = 0


    def dumpkvs(self):
        if self.counter % self.dilution == 0:
            for k, v in self.name2val.items():
                if type(v) is np.ndarray:
                    v = v.item()
                logger(k, v)
            super().dumpkvs()

        self.counter += 1
