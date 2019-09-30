from gym.core import Wrapper
from learning_and_planning.common_utils.data_reader import DataReader
from learning_and_planning.mcts.value import ValueDictionary
import numpy as np

roots = []
values = []

# Wrapper for a <single> SokobanEnv that reads from data
class SokobanReadFromData(Wrapper):
    def __init__(self, env, data, deadlock_reward=-10):
        super().__init__(env)
        if not values:  # to make self.values and self.roots 'static' variables
            for d in data:
                vf = ValueDictionary()
                vf.load_vf_for_root(d, compressed=True)
                values.append(vf)
                roots.append(vf.root)
        self.values = values
        self.roots = roots
        self.num_levels = len(data)
        self.current_room_id = -1
        self.deadlock_reward = deadlock_reward

    def step(self, action):
        obs, rew, don, info = self.env.step(action)
        info['deadlock'] = False
        state = self.env.clone_full_state()
        if self.values[self.current_room_id](states=state) == -np.inf:
            rew += self.deadlock_reward
            info['deadlock'] = True
        return obs, rew, don, info

    def reset(self, **kwargs):
        self.current_room_id = (self.current_room_id + 1) % self.num_levels
        root = self.roots[self.current_room_id]
        self.env.restore_full_state(root)
        # current_room_id enters info in SokobanEnv.step(), so has to be defined
        self.env.current_room_id = self.current_room_id
        return self.env.render(**kwargs)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    # INFO: makes the wrapper partially transparent (for vecenv)
    def __getattr__(self, name):
        if name in dir(self):
            func = getattr(self, name)
        else:
            func = getattr(self.env, name)
        return func



if __name__ == "__main__":
    from learning_and_planning.envs.sokoban_env_creator_lukasz import SokobanVecEnvCreatorLukasz
    from gym_sokoban.envs import SokobanEnv
    from PIL import Image
    import os
    import time

    def test_load_multiple_envs():
        env = SokobanEnv(dim_room=(8, 8),
                         num_boxes=2,
                         mode='tiny_rgb_array')
        data_files_prefix = os.path.join(os.environ.get('HOME'),
                                '',
                                '8x8x2_shards',
                                'shard')
        start = time.time()
        data = DataReader(data_files_prefix).load()
        print("Reading time: {}".format(time.time()-start))
        start = time.time()
        env = SokobanReadFromData(env, data)
        print("Sokoban wrapper time: {}".format(time.time()-start))
        start = time.time()
        env1 = SokobanReadFromData(env, data)
        print("Sokoban wrapper time: {}".format(time.time()-start))
        start = time.time()
        env2 = SokobanReadFromData(env, data)
        print("Sokoban wrapper time: {}".format(time.time()-start))
        print(id(env.values), id(env1.values), id(env2.values))
        for i in range(20):
            obs = env.reset()
            if i % 5 == 0:
                Image.fromarray(obs, "RGB").resize((40, 40)).show()

    def test_wrappers():
        from learning_and_planning.utils.wrappers import PlayWrapper, InfoDisplayWrapper, RewardPrinter
        env = SokobanEnv(dim_room=(8, 8),
                         num_boxes=2,
                         mode='rgb_array')
        # path to shard files
        data_files_prefix = os.path.join(os.environ.get('HOME'),
                                         '',
                                         '8x8x2_shards',
                                         'shard')
        data = DataReader(data_files_prefix).load()

        env = PlayWrapper(
            InfoDisplayWrapper(
                RewardPrinter(
                    SokobanReadFromData(env, data)
                ),
                augment_observations=True,
                min_text_area_width=500
            )
        )
        env.play()

    #test_load_multiple_envs()
    test_wrappers()
