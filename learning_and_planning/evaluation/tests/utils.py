from gym_sokoban.envs import SokobanEnv


class MockEnv:
    def __init__(self, make_done_after_num_steps_list, make_solved_list=(True,)):
        self.make_done_after_num_steps_list = list(make_done_after_num_steps_list)
        self.steps_to_done = None
        self.make_solved_list = list(make_solved_list)
        self.make_solved = None
        self.unwrapped = self
        self.max_steps = 20

    def reset(self):
        self.steps_to_done = self.make_done_after_num_steps_list.pop(0)
        self.make_solved = self.make_solved_list.pop(0)
        return None

    def step(self, action):
        self.steps_to_done -= 1
        if self.steps_to_done <= 0:
            done = True
            info = {'all_boxes_on_target': self.make_solved}
        else:
            done = False
            info = {}
        return None, None, done, info

    def render(self, *args, **kwargs):
        pass

    def close(self):
        pass


def get_test_env(seed=None):
    mode = 'tiny_rgb_array'
    _env = SokobanEnv(dim_room=(8, 8),
                      max_steps=100,
                      num_boxes=1,
                      mode=mode)
    if seed is not None:
        _env.seed(seed)
    return _env
