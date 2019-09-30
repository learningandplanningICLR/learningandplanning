from pathlib import Path

import attr
import gin
import gym
from gym import Wrapper

from ourlib.gym.video_recorder_wrapper import VideoRecorderWrapper
from dopamine.atari import preprocessing
from learning_and_planning.utils.gym_utils import RecordVideoTriggerEpisodeFreq


@gin.configurable
@attr.s
class AtariEnvCreator(object):
    video_directory = attr.ib()
    game_name = attr.ib()
    sticky_actions = attr.ib()

    def __call__(self, *args, **kwargs):
        print("GOOD", self.game_name, self.sticky_actions)
        def get_env_id(game_name, sticky_actions):
            game_version = 'v0' if sticky_actions else 'v4'
            full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
            return full_game_name

        env_id = get_env_id(self.game_name, self.sticky_actions)
        env = gym.make(env_id)
        # logger.info('env_id {}'.format(env_id))

        record_video_trigger = RecordVideoTriggerEpisodeFreq(episode_freq=500)

        # INFO: this is what dopamine does in create_atari_environement
        # They say:
        # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
        # handle this time limit internally instead, which lets us cap at 108k frames
        # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
        # restoring states.

        base_env = env.env

        env = VideoRecorderWrapper(base_env,
                                   directory=str(Path(self.video_directory) / 'videos'),
                                   record_video_trigger=record_video_trigger,
                                   video_length=2000000)

        class DopamineWrapper(Wrapper):
            def __init__(self, env, base_env):
                super().__init__(env)
                self.base_env = base_env

            @property
            def ale(self):
                return self.base_env.ale

        env = DopamineWrapper(env, base_env)
        env = preprocessing.AtariPreprocessing(env)

        return env
