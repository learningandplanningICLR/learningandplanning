import numpy as np
import attr
from typing import List


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    """Returns the current epsilon for the agent's epsilon-greedy policy.

  This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
  al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.

  Args:
    decay_period: float, the period over which epsilon is decayed.
    step: int, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before epsilon is decayed.
    epsilon: float, the final value to which to decay the epsilon parameter.

  Returns:
    A float, the current epsilon value computed according to the schedule.
  """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon)
    return epsilon + bonus


class LinearlyDecayingEpsilon:

    def __init__(self, lr, decay_period, warmup_steps, epsilon):
        self.epsilon = epsilon
        self.warmup_steps = warmup_steps
        self.decay_period = decay_period
        self.lr = lr

    def __call__(self, step):
        return self.lr*linearly_decaying_epsilon(self.decay_period, step, self.warmup_steps, self.epsilon)

    def __str__(self):
        return f"{self.lr}"

    def __repr__(self):
        return f"{self.lr}"


class StepDecay:

    def __init__(self, start_lr: float, steps: List[int], decay_factor: float = 5.):
        assert decay_factor >= 1.
        self.start_lr = start_lr
        self.decay_factor = decay_factor
        self.steps = steps

    def __call__(self, step):
        lr = self.start_lr
        for threshold in self.steps:
            if step > threshold:
                lr /= self.decay_factor
        return lr

    def __str__(self):
        return f"{self.start_lr}"

    def __repr__(self):
        return f"{self.start_lr}"


# import ipdb; ipdb.set_trace()
@attr.s
class HistoryElement(object):
    observation = attr.ib()
    reward = attr.ib()
    done = attr.ib()
    info = attr.ib()
