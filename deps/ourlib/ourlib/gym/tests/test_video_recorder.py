import tempfile

import gym

from ourlib.gym.video_recorder import VideoRecorderWrapper


def test_video_recorder():
    base_env = gym.make('Pong-v0')
    directory = tempfile.mkdtemp()

    wrapped_env = VideoRecorderWrapper(base_env,
                                       directory=directory,
                                       record_video_trigger=lambda step_id: True,
                                       video_length=20)

    env = wrapped_env

    env.reset()
    idx = 0
    while True:
        action = env.action_space.sample()
        ob, reward, done, _ = env.step(action)
        idx += 1
        if done:
            break
    print(idx)
    print(directory)
    env.close()
