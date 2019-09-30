import tempfile
from collections import defaultdict
from typing import Callable
import attr
import gin
from deprecated import deprecated
from gym import Wrapper
from tensorflow.python.summary.writer.writer import FileWriter

from ourlib.gym.utils import play_random_till_episode_end, add_creator
from ourlib.summary.summary_helper import SummaryHelper
from gym_sokoban.envs import SokobanEnv

#from learning_and_planning.common_utils.subproc_vec_env_save_restore import SubprocVecEnvCloneRestore
from learning_and_planning.common_utils.vec_env_clone_restore import VecEnvCloneRestore
import numpy as np
import importlib


@add_creator
class EpisodeHistoryCallbackWrapper(Wrapper):
    def __init__(self, env, episode_history_callbacks):
        super().__init__(env)
        self.episode_history_callbacks = episode_history_callbacks
        self.history = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.history.append((obs, reward, done, info))
        return obs, reward, done, info

    def reset(self):
        if self.history is not None:
            for episode_history_callback in self.episode_history_callbacks:
                episode_history_callback(self.history)

        obs = self.env.reset()
        self.history = [(obs, None, None, None)]
        return obs

@add_creator
class BetterThanThresholdCurriculumSetter:

    def __init__(self, env, smooth_coeff, threshold, initial_value,
                 statistics_to_follow, change_field_by, running_average_drop_on_increase,
                 field_to_modify, max_field_value, summary_helper:SummaryHelper):
        self.smooth_coeff = smooth_coeff
        self.threshold = threshold
        self.initial_value = initial_value
        self.statistics_to_follow = statistics_to_follow
        self.running_average = self.initial_value
        self.change_field_by = change_field_by
        self.running_average_drop_on_increase = running_average_drop_on_increase
        self.field_to_modify = field_to_modify
        self.env = env
        self.max_field_value = max_field_value
        self.summary_helper = summary_helper

    def recalculate_curriculum(self, name, y, freq, global_step):

        self.running_average = self.smooth_coeff*self.running_average + (1-self.smooth_coeff)*y

        if name == self.statistics_to_follow:
            self.summary_helper.add_simple_summary("curriculum {}".format(self.rank), getattr(self.env, self.field_to_modify),
                                                   freq=1, global_step=global_step)
            self.summary_helper.add_simple_summary("curriculum_running_average {}".format(self.rank), self.running_average,
                                                   freq=1, global_step=global_step)

        if self.running_average > self.threshold:
            self.running_average -= self.running_average_drop_on_increase
            old_val = getattr(self.env, self.field_to_modify)
            new_value = old_val+self.change_field_by
            new_value = new_value if new_value<=self.max_field_value else old_val
            setattr(self.env, self.field_to_modify, new_value)


class EpisodeHistorySummarizer(object):
    def __init__(self, summary_helper: SummaryHelper, curriculum_setter: BetterThanThresholdCurriculumSetter,
                 freq=20, global_step_getter_fn: Callable = None):

        self.summary_helper = summary_helper
        self.freq = freq
        self.global_step_getter_fn = global_step_getter_fn
        self.my_global_step = 0
        self.curriculum_setter = curriculum_setter

    def __call__(self, history):
        aux_rewards = defaultdict(list)

        for obs, reward, done, info in history:

            if info and 'aux_rewards' in info:  # e.g. 'solved' in 'aux_rewards'
                for key, value in info['aux_rewards'].items():
                    aux_rewards[key].append(value)

        for aux_reward_name, rewards in aux_rewards.items():
            # sum
            name = 'aux_reward/{}/sum'.format(aux_reward_name)
            y = np.sum(rewards)
            global_step = self.get_global_step()
            # print('name = {}, y = {}, global_step = {}'.format(name, y, global_step))
            self.summary_helper.add_simple_summary(name, y=y, freq=self.freq, global_step=global_step)
            if self.curriculum_setter:
                self.curriculum_setter.recalculate_curriculum(name, y=y,
                                                              freq=self.freq, global_step=global_step)

            # mean
            name = 'aux_reward/{}/mean'.format(aux_reward_name)
            y = np.mean(rewards)
            global_step = self.get_global_step()
            # print('name = {}, y = {}, global_step = {}'.format(name, y, global_step))
            self.summary_helper.add_simple_summary(name, y=y, freq=self.freq, global_step=global_step)
            if self.curriculum_setter:
                self.curriculum_setter.recalculate_curriculum(name, y=y,
                                                              freq=self.freq, global_step=global_step)

        self.my_global_step += 1

    def get_global_step(self):
        if self.global_step_getter_fn is None:
            return self.my_global_step
        else:
            return self.global_step_getter_fn()


def get_callable(callable_full_name):
    idx = callable_full_name.rfind(":")
    module_name = callable_full_name[:idx]
    callable_name = callable_full_name[idx + 1:]
    env_module = importlib.import_module(module_name)
    _callable = getattr(env_module, callable_name)

    return _callable


@gin.configurable
def get_env_creator(env_callable_name, num_envs=1, **kwargs):
    _callable = None
    if callable(env_callable_name):
        _callable = env_callable_name
    if type(env_callable_name) == str:
        _callable = get_callable(env_callable_name)

    def _create_env():
        return _callable(**kwargs)
    if num_envs == 1:
        return _create_env
    else:
        #return lambda:  SubprocVecEnvCloneRestore([_create_env for _ in range(num_envs)])
        return lambda:  VecEnvCloneRestore(_create_env, nenvs = num_envs)


@gin.configurable
@attr.s
class SokobanEnvCreator(object):
    dim_room = attr.ib()
    max_steps = attr.ib()
    num_boxes = attr.ib()
    seed = attr.ib()
    max_distinct_rooms = attr.ib()
    mode = attr.ib()  # 'rgb_array', 'tiny_rgb_array'
    game_mode = attr.ib()
    num_gen_steps = attr.ib()

    @deprecated(reason="You should use get_env_creator")
    def __call__(self, *args, **kwargs):
        env = SokobanEnv(dim_room=self.dim_room, max_steps=self.max_steps, num_boxes=self.num_boxes,
                         mode=self.mode, max_distinct_rooms=self.max_distinct_rooms, game_mode=self.game_mode,
                         num_gen_steps=self.num_gen_steps)
        env.seed(self.seed)

        return env

@gin.configurable
@attr.s
class SokobanSerialVecEnvCreator(object):
    dim_room = attr.ib()
    # max_steps = attr.ib()
    num_boxes = attr.ib()
    mode = attr.ib()
    num_env = attr.ib()
    game_mode = attr.ib(default='NoAlice')

    @deprecated(reason="You should use get_env_creator")
    def __call__(self, seed=None, data=None, deadlock_reward=-10, *args, **kwargs):
        def make_env(*args, **kwargs):
            def _thunk():
                base_env = SokobanEnv(dim_room=self.dim_room,  # max_steps=self.max_steps,
                                      num_boxes=self.num_boxes, mode=self.mode, game_mode=self.game_mode)
                base_env.seed(seed)
                return base_env

            return _thunk

        return VecEnvCloneRestore(make_env(*args, **kwargs), nenvs = self.num_env)


@gin.configurable
class CurriculumSetterCallback(object):

    def __init__(self, env, smooth_coeff, threshold, initial_value,
                 statistics_to_follow, change_field_by, running_average_drop_on_increase,
                 field_to_modify, max_field_value, field_intial_value):
        self.smooth_coeff = smooth_coeff
        self.threshold = threshold
        self.initial_value = initial_value
        self.statistics_to_follow = statistics_to_follow
        self.running_average = self.initial_value
        self.change_field_by = change_field_by
        self.running_average_drop_on_increase = running_average_drop_on_increase
        self.field_to_modify = field_to_modify
        self.env = env
        self.max_field_value = max_field_value
        self.field_value = field_intial_value

        self.env.pm_setattr(self.field_to_modify, self.field_value)

    def invoke(self, logger=None, **kwargs):
        y = kwargs[self.statistics_to_follow]
        self.running_average = self.smooth_coeff * self.running_average + (1 - self.smooth_coeff) * y

        if self.running_average > self.threshold:
            self.running_average -= self.running_average_drop_on_increase
            new_value = self.field_value + self.change_field_by
            self.field_value = new_value if new_value <= self.max_field_value \
                else self.field_value
            self.env.pm_setattr(self.field_to_modify, self.field_value)
        if logger:
            logger.record_tabular("curriculum_{}".format(self.field_to_modify), self.field_value)
            logger.record_tabular("curriculum_running_av_{}".format(self.statistics_to_follow), self.running_average)




def test_EpisodeHistorySummarizer():
    file_writer = FileWriter(logdir=tempfile.mktemp())
    summary_helper = SummaryHelper(file_writer)
    env_creator = SokobanEnvCreator(max_steps=120,
                                    num_boxes=1, max_distinct_rooms=2, mode='one_hot',
                                    dim_room=(10, 10),
                                    seed=2)
    env = env_creator()

    print(dir(EpisodeHistoryCallbackWrapper))
    creators = [
        EpisodeHistoryCallbackWrapper.create_creator([EpisodeHistorySummarizer(summary_helper, freq=1)])
    ]

    for creator in creators:
        env = creator(env)

    env.reset()
    play_random_till_episode_end(env)
    env.reset()


def test_env_creator():
    env_creator = get_env_creator(env_module_name="gym_sokoban.envs",
                                  env_callable_name="SokobanEnv", dim_room=(8, 8), num_boxes=2)
    env = env_creator()
    print(env)

    env_creator = get_env_creator(env_module_name="gym_sokoban.envs",
                                  env_callable_name="SokobanEnv", dim_room=(8, 8),
                                  num_boxes=2, num_envs=4)
    env = env_creator()
    print(env)


if __name__ == '__main__':
    test_EpisodeHistorySummarizer()
    test_env_creator()
