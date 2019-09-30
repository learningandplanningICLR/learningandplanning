from pathlib import Path
from time import sleep

import attr
import gin
import gym

from ourlib.gym.utils import CallbackWrapper
from ourlib.gym.video_recorder_wrapper import VideoRecorderWrapper
from learning_and_planning.utils.gym_utils import RecordVideoTriggerEpisodeFreq


@gin.configurable
@attr.s
class GymEnvCreator(object):
    video_directory = attr.ib()
    env_id = attr.ib()

    def __call__(self, *args, **kwargs):
        env = gym.make(self.env_id)
        if self.video_directory:
            record_video_trigger = RecordVideoTriggerEpisodeFreq(episode_freq=10000)
            env = VideoRecorderWrapper(env,
                                       directory=str(Path(self.video_directory) / 'videos'),
                                       record_video_trigger=record_video_trigger,
                                       video_length=2000000)

        # def step(ob, reward, done, info):
        #     sleep(0.002)
        #     return ob, reward, done, info
        #
        # env = CallbackWrapper(env, step_callback=step)

        return env
