import os
from typing import Union, List

from PIL import Image
from gym import Wrapper

from pathlib2 import Path

from ourlib.gym.utils import add_creator
from ourlib.gym.video_recorder import AwarelibVideoRecorder
from ourlib.image.utils import write_text_on_image, PIL_to_ndarray, concatenate_images, add_text_bottom
from ourlib.logger import logger

class RecordVideoTrigger(object):
    def __call__(self, step_id, episode_id):
        raise NotImplementedError


class AlwaysTrueRecordVideoTrigger(RecordVideoTrigger):
    def __call__(self, step_id, episode_id):
        return True


@add_creator
class VideoRecorderWrapper(Wrapper):
    """
    Wrap Env to record rendered image as mp4 video.
    NOTE(): this is almost entirely copied from vec_video_recorder.py from baselines
    """

    def __init__(self, env,
                 directory='/tmp/gym_videos/',
                 record_video_trigger=AlwaysTrueRecordVideoTrigger(),
                 video_length=2000, summary_helper=None):

        super(VideoRecorderWrapper, self).__init__(env)

        self.record_video_trigger = record_video_trigger
        self.video_recorder = None

        self.directory = os.path.abspath(directory)
        if not Path(self.directory).exists():
            Path(self.directory).mkdir(parents=True)

        self.file_prefix = "env"
        self.file_infix = '{}'.format(os.getpid())
        self.step_id = 0
        self.episode_id = -1
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0
        self.sum_rew = 0
        self.last_rew = None

        self.reset_text_to_add()
        self.observations = []
        self.summary_helper = summary_helper
        self.last_info = None
        self.neptune_steps = 0

    def reset(self):
        self.dump_to_neptune()
        self.step_id = 0
        self.sum_rew = 0
        self.episode_id += 1
        self.last_info = None
        obs = self.env.reset()

        self.close_video_recorder()

        return obs

    def dump_to_neptune(self):
        if self.last_info is None:
            return
        if "aux_rewards" in self.last_info:
            channel_name = "solved" if self.last_info['aux_rewards']['solved'] else "failed"
        else:
            channel_name = "game"
        if self.summary_helper:
            for ob in self.observations:
                self.neptune_steps += 1
                self.summary_helper.add_image_summary(channel_name,
                                                      Image.fromarray(ob),
                                                      global_step=self.neptune_steps)
        self.observations = []


    def start_video_recorder(self, transform_frame_fn=None):
        self.close_video_recorder()

        path = os.path.join(self.directory, '{}.video.{}.video_{:06}_{:010}_{}.mp4'.format(
            self.file_prefix, self.file_infix, self.episode_id, self.step_id, self.sum_rew))

        self.video_recorder = AwarelibVideoRecorder(
                env=self.env,
                path=path,
                metadata={'step_id': self.step_id, 'epsiode_id': self.episode_id}
                )


        self.video_recorder.capture_frame(transform_frame_fn=transform_frame_fn)
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self):
        return self.record_video_trigger(step_id=self.step_id, episode_id=self.episode_id)

    def step(self, action):
        # We first render the frame, with the action we got
        self.add_text('action = {}'.format(action))
        self.add_text('step_id = {}'.format(self.step_id))
        self.add_text('last_rew = {}'.format(self.last_rew))
        self.add_text('sum_rew = {0:.1f}'.format(self.sum_rew))

        if (self.last_info is not None and isinstance(self.last_info, dict) and
            'text' in self.last_info):
            self.add_text(self.last_info['text'])



        def transform_frame_fn(frame):
                return add_text_bottom(frame, self.text_to_add)

        if self.recording:
            self.video_recorder.capture_frame(transform_frame_fn=transform_frame_fn)
            self.reset_text_to_add()
            self.observations.append(self.env.render('rgb_array'))

            self.recorded_frames += 1
            if self.recorded_frames > self.video_length:
                logger.info("Saving video to ", self.video_recorder.path)
                self.close_video_recorder()
        elif self._video_enabled():
            self.reset_text_to_add()
            self.start_video_recorder(transform_frame_fn=transform_frame_fn)
            self.observations.append(self.env.render('rgb_array'))

        ob, rew, done, info = self.env.step(action)
        self.step_id += 1
        self.sum_rew += rew
        self.last_rew = rew
        self.last_info = info

        return ob, rew, done, info

    def reset_text_to_add(self):
        self.text_to_add = []

    def add_text(self, text: Union[str, List[str]]):
        if isinstance(text, list):
            self.text_to_add += text
        else:
            self.text_to_add.append(text)

    def close_video_recorder(self):
        if self.recording:
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 0

    def close(self):
        super(VideoRecorderWrapper, self).close()
        self.close_video_recorder()

    def __del__(self):
        self.close()

