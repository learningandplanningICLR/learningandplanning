import json
import logging
import os

from gym.utils import atomic_write
from gym.utils.json_utils import json_encode_np
from gym.wrappers import Monitor
from gym.wrappers.monitoring.stats_recorder import StatsRecorder

from learning_and_planning.utils.wrappers import SokobanTransparentWrapperMixin

logger = logging.getLogger(__name__)


class EvaluationStatsRecorder(StatsRecorder):
    """ Records statistics about solved levels. """
    def __init__(self, directory, file_prefix, autoreset=False, env_id=None):
        super().__init__(directory, file_prefix, autoreset, env_id)
        self.last_solved = None
        self.solved_flags = []

    def after_step(self, observation, reward, done, info):
        if done:
            self.last_solved = info.get('aux_rewards', {}).get('solved', False)
        super().after_step(observation, reward, done, info)

    def save_complete(self):
        super().save_complete()
        self.solved_flags.append(self.last_solved)

    def flush(self):
        if self.closed:
            return

        with atomic_write.atomic_write(self.path) as f:
            json.dump({
                'initial_reset_timestamp': self.initial_reset_timestamp,
                'timestamps': self.timestamps,
                'episode_lengths': self.episode_lengths,
                'episode_rewards': self.episode_rewards,
                'episode_types': self.episode_types,
                'solved_flags': self.solved_flags,
            }, f, default=json_encode_np)


class EvaluationMonitor(Monitor, SokobanTransparentWrapperMixin):
    """ Monitor with custom StatsRecorder and allowing to move videos to subdirectories. """
    def _start(self, directory, video_callable=None, force=False, resume=False, write_upon_reset=False, uid=None,
               mode=None):
        super()._start(directory, video_callable, force, resume, write_upon_reset, uid, mode)
        # Custom stats recorder
        self.stats_recorder = EvaluationStatsRecorder(
            directory,
            '{}.episode_batch.{}'.format(self.file_prefix, self.file_infix),
            autoreset=self.env_semantics_autoreset,
            env_id=None
        )
        self.solved_dir = os.path.join(directory, 'solved')
        self.unsolved_dir = os.path.join(directory, 'unsolved')
        os.makedirs(self.solved_dir, exist_ok=True)
        os.makedirs(self.unsolved_dir, exist_ok=True)

        # Prevent bogus `close` behaviour`
        self._monitor = self._monitor_id

    def _after_reset(self, observation):
        super()._after_reset(observation)
        self.stats_recorder.last_solved = None

    def _close_video_recorder(self):
        """ Distribute video after closing. """
        super()._close_video_recorder()
        self._distribute_video(self.video_recorder.path)

    def _distribute_video(self, video_path):
        """ Move video to subdirectory. """
        filename = os.path.basename(video_path)
        new_dir = self._distribute_directory()
        new_path = os.path.join(new_dir, filename)
        os.rename(video_path, new_path)
        return new_path

    def _distribute_directory(self):
        """ Choose directory to move the video to. """
        return self.solved_dir if self.stats_recorder.last_solved else self.unsolved_dir
