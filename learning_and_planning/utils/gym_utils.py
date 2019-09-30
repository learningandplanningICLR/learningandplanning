import math
from collections.abc import Mapping

from ourlib.gym.video_recorder_wrapper import RecordVideoTrigger


class RecordVideoTriggerEpisodeFreq(RecordVideoTrigger):
    def __init__(self, episode_freq=1000):
        self.episode_freq = episode_freq
        self.last_recorded_episode_id = -math.inf

    def __call__(self, step_id, episode_id):
        if episode_id > self.last_recorded_episode_id + self.episode_freq:
            # print(episode_id, self.last_recorded_episode_id + self.episode_freq)
            self.last_recorded_episode_id = episode_id
            return True
        else:
            return False


class KeysToActionMapping(Mapping):
    """
    Acts as if `mapping` was a defaultdict(lambda: noop_action),
    but returns noop_action even for `get` with a default.
    See example below for explanation.

    Examples:
    >>> m = KeysToActionMapping({0: 0, 1: 1}, noop_action=-1)
    >>> m.get(0)
    0
    >>> m.get(10)  # defaultdict(lambda: -1) would work here too
    -1
    >>> m.get(10, 10)  # defaultdict(lambda: -1) would return 10 here
    -1
    """
    def __init__(self, mapping, noop_action=0):
        self.mapping = mapping
        self.noop_action = noop_action

    def __getitem__(self, k):
        return self.mapping.get(k, self.noop_action)

    def __iter__(self):
        return self.mapping.__iter__()

    def __len__(self):
        return self.mapping.__len__()