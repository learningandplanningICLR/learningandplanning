from time import sleep

import gym

from ourlib.gym.utils import CallbackWrapper, play_random_till_episode_end


def test_callback_wrapper():
    env = gym.make('Pong-v0')

    def step(ob, reward, done, info):
        return ob, reward, done, info

    env = CallbackWrapper(env, step_callback=step)
    env.reset()
    play_random_till_episode_end(env)
    env.close()


if __name__ == '__main__':
    test_callback_wrapper()
