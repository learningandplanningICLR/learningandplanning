from typing import Callable, List

from gym import Wrapper


def play_random_till_episode_end(env, max_steps=10000000000):
    idx = 0
    while True:
        action = env.action_space.sample()
        ob, reward, done, _ = env.step(action)
        idx += 1
        if done or idx > max_steps:
            break

play_one_episode = play_random_till_episode_end

def add_creator(cls):
    '''This decorator add @classmethods `create_factory` will take same arguments as
    the __init__ method of the class and will return a function will will take one argument `env`
    and the call __init__methods of the class, add the arguments passed to create_factory.

    This is useful when we want to create the pipeline of environment wrappers, but for
    some reason we have to create the actual environment later.

    '''
    def create_creator(*args, **kwargs):
        def creator_fn(env):
            return cls(env, *args, **kwargs)

        return creator_fn

    cls.create_creator = create_creator
    return cls


def merge_env_creators(creators: List[Callable]):
    def merged_env_creator_fn():
        print('Called!!!')
        env = creators[0]()
        for creator in creators[1:]:
            env = creator(env)
        print('Created', env)
        return env

    return merged_env_creator_fn


class CallbackWrapper(Wrapper):
    def __init__(self, env, reset_callback=None, step_callback=None):
        super().__init__(env)
        self.reset_callback = reset_callback
        self.step_callback = step_callback

    def step(self, action):
        res = super().step(action)
        if self.step_callback is not None:
            res = self.step_callback(*res)
        return res

    def reset(self, **kwargs):
        res = super().reset(**kwargs)
        if self.reset_callback is not None:
            res = self.reset_callback_callback(res)
        return res
