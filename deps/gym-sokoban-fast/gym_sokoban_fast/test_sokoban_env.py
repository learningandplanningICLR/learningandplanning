import unittest
import numpy as np

from gym_sokoban_fast.sokoban_env_fast import SokobanEnvFast


class TestSokobanFastEnvSeed(unittest.TestCase):

    def test_seed_state(self, dim_room=(10,10), max_steps=100, num_boxes=4, mode='one_hot'):
        env = SokobanEnvFast(
            dim_room=dim_room, max_steps=max_steps, num_boxes=num_boxes, mode=mode
        )
        env.reset()
        seed = np.random.randint(0,100)
        env.seed(seed)
        env.reset()
        state = env.clone_full_state().one_hot

        for _ in range(10):
            env.seed(seed)
            env.reset()
            new_state = env.clone_full_state().one_hot
            self.assertTrue((new_state == state).all())

    def test_seed_observation(self, dim_room=(10,10), max_steps=100, num_boxes=4, mode='one_hot'):
        env = SokobanEnvFast(
            dim_room=dim_room, max_steps=max_steps, num_boxes=num_boxes, mode=mode
        )
        env.reset()
        seed = np.random.randint(0,100)
        env.seed(seed)
        ob = env.reset()

        for _ in range(10):
            env.seed(seed)
            new_ob = env.reset()
            self.assertTrue((new_ob == ob).all())



if __name__ == '__main__':
    unittest.main()
