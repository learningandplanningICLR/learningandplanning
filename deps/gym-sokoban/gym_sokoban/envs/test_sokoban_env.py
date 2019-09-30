import pyglet
from ourlib.gym.video_recorder_wrapper import VideoRecorderWrapper
from ourlib.summary.summary_helper import SummaryHelper
from numpy.testing import assert_array_equal

from ourlib.image.utils import show_numpy_img_sxiv
import gym
from gym_sokoban.envs import render_utils, SokobanEnv, OneHotTypeSets
from gym.utils import seeding
from gym.envs import register
import numpy as np

import collections
import time
import tensorflow as tf

from learning_and_planning.envs.sokoban_env_creator import BetterThanThresholdCurriculumSetter, get_env_creator, \
    EpisodeHistoryCallbackWrapper, EpisodeHistorySummarizer
from learning_and_planning.mcts.value import ValuePerfect
from learning_and_planning.utils.gym_utils import RecordVideoTriggerEpisodeFreq
from learning_and_planning.utils.wrappers import PlayWrapper, InfoDisplayWrapper, RewardPrinter, RestoreStateWrapper
from learning_and_planning.evaluation import monitor


def print_state(state, dim_room=(10,10)):
    state = state[:-1]
    state = np.array(state)
    split = len(state) // 2
    room_state = state[split:].reshape((dim_room[0], dim_room[1]))
    print(room_state)

def create_env(seed, dim_room=(13,13), num_boxes=5):
    env = SokobanEnv(dim_room=dim_room, max_steps=100, num_boxes=num_boxes,
                     mode='rgb_array', max_distinct_rooms=10)
    env.seed(seed)
    return env


# TODO: Make it pass.
#def test_tiny_rgb_mode(scale=20):
#    dim_room = (5, 5)
#    env = SokobanEnv(dim_room=dim_room, max_steps=100, num_boxes=2,
#                     mode='tiny_rgb_array', max_distinct_rooms=10)
#    _ = env.reset()
#    obs = env.render(scale=scale)
#    print(obs)
#    # assert obs.shape == dim_room + (7,)
#    assert obs.dtype == np.uint8
#    show_numpy_img_sxiv(obs)
#    print(obs.shape)


def test_one_hot_mode():
    dim_room = (10, 10)
    env = SokobanEnv(dim_room=dim_room, max_steps=100, num_boxes=2,
                     mode='one_hot', max_distinct_rooms=10)
    obs = env.reset()
    assert obs.shape == dim_room + (7,)
    assert obs.dtype == np.uint8
    print(obs.shape)
    # print(obs)


# TODO: Make it pass.
#def test_one_hot_flatten():
#    dim_room = (10, 10)
#    env = SokobanEnv(dim_room=dim_room, max_steps=100, num_boxes=2,
#                     mode='one_hot_flatten', max_distinct_rooms=10)
#    obs = env.reset()
#    assert obs.shape == dim_room[0] * dim_room[1] * 7
#    assert obs.dtype == np.uint8
#    print(obs.shape)
#    # print(obs)


def test_seed():
    seed = 20

    env1 = create_env(seed=seed)
    env2 = create_env(seed=seed)

    obs1 = env1.reset()

    show_numpy_img_sxiv(obs1)
    obs2 = env2.reset()
    show_numpy_img_sxiv(obs2)

    assert_array_equal(obs1, obs2)

# TODO: Make it pass.
#def test_seed_2(mode='tiny_rgb_array', dim_room=(10,10), seed=None):
#    scale = 20 if 'tiny' in mode else 1
#    env1 = create_env(seed, dim_room)
#    env1.reset()
#
#    initial_state = env1.clone_full_state()
#    print_state(initial_state, dim_room)
#    obs = env1.render(mode, scale=scale)
#    frames = [obs]
#    actions = []
#    rng = np.random.RandomState()
#    rng.seed(seed)
#    for _ in range(rng.randint(10)):
#        a = rng.randint(env1.action_space.n)
#        actions.append(a)
#        _, _, done, _ = env1.step(a)
#        obs = env1.render(mode, scale=scale)
#        frames.append(obs)
#        if done:
#            break
#    end_state = env1.clone_full_state()
#    image = np.hstack(frames)
#    from PIL import Image
#    Image.fromarray(image, "RGB").show()
#
#    env2 = create_env(seed, dim_room)
#    env2.reset()
#    env2.restore_full_state(initial_state)
#    obs = env2.render(mode, scale=scale)
#    frames = [obs]
#    for a in actions:
#        env2.step(a)
#        obs = env2.render(mode, scale=scale)
#        frames.append(obs)
#    image = np.hstack(frames)
#    Image.fromarray(image, "RGB").show()
#    #show_numpy_img_sxiv(image)
#    print((end_state == env2.clone_full_state()).all())

def stress_test(dim=(13,13), num_boxes=5, num_runs=100, seed=None):
    if not seed:
      _, seed = seeding.np_random(None)
    env = create_env(seed=seed, dim_room=dim, num_boxes=num_boxes)
    start = time.clock()
    for _ in range(num_runs):
        env.reset()
        state = env.clone_full_state()
        print_state(state, dim_room=dim)
    end = time.clock()
    print((end-start)/num_runs)

def test_recover(dim=(13,13), num_boxes=5, mode='rgb_array', seed=None):
    if not seed:
        _, seed = seeding.np_random(None)
    env = SokobanEnv(dim_room=dim, max_steps=100, num_boxes=num_boxes,
                     mode=mode, max_distinct_rooms=10)
    env.seed(seed)
    env.reset()
    obs = env.render()
    state = env.clone_full_state()
    print(state == env.recover_state(obs))

def test_img():
    env = SokobanEnv(dim_room=(10,10), max_steps=100, num_boxes=4,
                     mode='rgb_array', max_distinct_rooms=10)
    from PIL import Image
    for i in range(10):
        env.reset()
        img = env.render()
        Image.fromarray(img, "RGB").save("{}.png".format(i))

def test_seed(dim=(13,13), num_boxes=5, mode='rgb_array', seed=None):
    from ctypes import c_uint
    if not seed:
        _, seed = seeding.np_random(None)
    env = SokobanEnv(dim_room=dim, max_steps=100, num_boxes=num_boxes,
                     mode='rgb_array')
    env.seed(seed)
    print("Seed: {}".format(np.uint32(c_uint(seed))))
    from PIL import Image
    env.reset()
    img = env.render()
    Image.fromarray(img, "RGB").resize((200, 200)).show()

def test_type_counts(dim_room=(13, 13), num_boxes=4):
    env = SokobanEnv(
        dim_room=dim_room, max_steps=100, num_boxes=num_boxes, mode='one_hot'
    )
    ob = env.reset()
    type_counter = collections.Counter(
        np.reshape(np.argmax(ob, axis=2), newshape=(-1,))
    )
    def assert_type_count(type_set, number):
        assert sum(type_counter[type] for type in type_set) == number
    assert_type_count(OneHotTypeSets.player, 1)
    assert_type_count(OneHotTypeSets.box, num_boxes)
    assert_type_count(OneHotTypeSets.target, num_boxes)

def test_curriculum(dim=(13,13), num_boxes=5, mode='rgb_array', seed=None, curriculum=300):
    from ctypes import c_uint
    if not seed:
        _, seed = seeding.np_random(None)
    env = SokobanEnv(dim_room=dim, max_steps=100, num_boxes=num_boxes,
                     mode=mode, curriculum=curriculum)
    env.seed(seed)
    print("Seed: {}".format(np.uint32(c_uint(seed))))
    from PIL import Image
    env.reset()
    img = env.render()
    Image.fromarray(img, "RGB").resize((200, 200)).show()

def test_curriculum_2(dim=(8,8), num_boxes=2, mode='rgb_array', seed=None, video_recording=False, curriculum=300):

    params = {
        'curriculum_smooth_coeff': [.99],
        'curriculum_threshold': [0.85],
        'curriculum_initial_value': [0.0],
        'curriculum_statistics_to_follow': ["aux_reward/solved/sum"],
        'curriculum_change_field_by': [1],
        'curriculum_running_average_drop_on_increase': [0.1],
        'curriculum_field_to_modify': ["num_gen_steps"],
        'curriculum_max_field_value': [23]
    }

    exp_dir_path = '.'
    summary_writer = tf.summary.FileWriter(exp_dir_path)
    summary_helper = SummaryHelper(summary_writer)

    curriculum_kwargs = {}
    for param_name in params:
        curriculum_kwargs[param_name[len("curriculum_"):]] = params[param_name]


    curriculum_setter_fn = \
        BetterThanThresholdCurriculumSetter.create_creator(summary_helper=summary_helper,
                                                           **curriculum_kwargs)

    if video_recording:
        video_directory = 'videos'
    else:
        video_directory = None

    record_video_trigger = RecordVideoTriggerEpisodeFreq(episode_freq=500)

    def env_creator():
        # This is awkward, but gym adds some magic required for wrappers
        env_creator = get_env_creator(env_callable_name="gym_sokoban.envs:SokobanEnv", seed=seed)

        register("BareEnv-v0", entry_point=env_creator)
        env = gym.make("BareEnv-v0")

        curriculum_setter = curriculum_setter_fn(env)
        env = EpisodeHistoryCallbackWrapper(env,[EpisodeHistorySummarizer(summary_helper,
                                                                          curriculum_setter, freq=20)])
        if video_directory:
            env = VideoRecorderWrapper(env, directory=video_directory,
                                       record_video_trigger=record_video_trigger,
                                       video_length=2000000, summary_helper=summary_helper)
        return env

    #This is required to pass evironment to baselines
    register('TmpEnvAwarelab-v0', entry_point=env_creator)

    env = gym.make('TmpEnvAwarelab-v0')
    env.reset()

    from PIL import Image
    img = env.render(mode='rgb_array')
    Image.fromarray(img, "RGB").resize((200, 200)).show()

def serialize_game(game, types, buf_size):
    _game = [x.flatten() for x in map(np.stack, zip(*game))]
    padding = [np.zeros(size-len(x), dtype=type).tobytes()
               for type, size, x in zip(types, buf_size, _game)]
    _game = [x.tobytes() for x in _game]
    lengths = [np.array(len(x), dtype=np.int32).tobytes() for x in _game]  # actual size of game
    _game = [x+p for x, p in zip(_game, padding)]
    _game = lengths + _game
    _game = b''.join(_game)
    return _game

    # INFO: assumes self._shape not None (i.e. self.to_storage() was called earlier
def deserialize_game(game_serialized, buf_size, shapes, types):
    int_size = len(np.array(0, dtype=np.int32).tobytes())
    ticks = np.cumsum([0, int_size, int_size, int_size])

    lengths = [np.frombuffer(game_serialized[start:end], dtype=np.int32).item()
               for start, end in zip(ticks[:-1], ticks[1:])]  # actual size of game
    game_serialized = game_serialized[ticks[-1]:]
    chunks = [bs * len(np.array(0, dtype=type).tobytes())
              for bs, type in zip(buf_size, types)]
    chunks = np.cumsum([0] + list(chunks))
    data = [game_serialized[start:end]
           for start, end in zip(chunks[:-1], chunks[1:])]  # get chunks according to buf_size
    data = [d[:b] for d, b in zip(data, lengths)]
    data = [np.frombuffer(d, dtype=type) for d, type in zip(data, types)]
    data = [np.squeeze(d.reshape((-1, *shape))) for d, shape in zip(data, shapes)]
    data = [(s, o, v) for s, o, v in zip(data[0], data[1], data[2])]
    return data
    #return list(zip(data))

def test_serialization(dim=(8, 8), num_boxes=1, mode='rgb_array', seed=None, curriculum=300):
    from ctypes import c_uint
    if not seed:
        _, seed = seeding.np_random(None)
    env = SokobanEnv(dim_room=dim, max_steps=100, num_boxes=num_boxes,
                     mode=mode, curriculum=curriculum)
    env.seed(seed)
    env.reset()

    state = env.clone_full_state()
    obs = env.render(mode='rgb_array')
    value = np.float32(5.0)

    shapes = (state.shape, obs.shape, (1,))
    type = (state.dtype, obs.dtype, np.float32)
    buf_size = env.max_steps * np.array([np.prod(x) for x in shapes])

    game = [(state, obs, value), (state, obs, value)]
    serial = serialize_game(game, type, buf_size)
    zz = np.frombuffer(serial, dtype=np.uint8)

    dgame = deserialize_game(serial, buf_size, shapes, type)

    return [[(i==j).all() for i, j in zip(a,b)] for a, b in zip(game, dgame)]


def test_room_to_binary_map_and_back():
    env = SokobanEnv()
    for _ in range(100):
        env.reset()
        flat_state = env.clone_full_state()
        (state, structure) = render_utils.get_room_state_and_structure(
            flat_state, env.dim_room
        )
        room = render_utils.make_standalone_state(state, structure)
        binary_map = render_utils.room_to_binary_map(room)
        converted_room = render_utils.binary_map_to_room(binary_map)
        assert (converted_room == room).all()

def test_draw_hard_level():
    room_state = np.array(\
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 3, 0, 3, 0, 3, 1, 0],
         [0, 0, 1, 0, 5, 4, 1, 0, 0],
         [0, 0, 1, 0, 1, 0, 2, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    room_fixed = np.array(\
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 2, 0, 2, 0, 2, 1, 0],
         [0, 0, 1, 0, 1, 1, 1, 0, 0],
         [0, 0, 1, 0, 1, 0, 2, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    #room_state = np.array( \
    #    [[0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 2, 2, 2, 2, 0, 0, 0],
    #     [0, 0, 4, 4, 4, 4, 0, 0, 0],
    #     [0, 0, 5, 1, 1, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    #room_fixed = np.array( \
    #    [[0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 2, 2, 2, 2, 0, 0, 0],
    #     [0, 0, 1, 1, 1, 1, 0, 0, 0],
    #     [0, 0, 1, 1, 1, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    state0 = np.concatenate([room_fixed.flatten(), room_state.flatten(), np.array([0])])
    state = state0
    #env = PlayWrapper(
    #            SokobanEnv(dim_room=(9, 9), num_boxes=4, game_mode="NoAlice", mode="rgb_array"))

    env_kwarg = {
        "env_callable_name": "gym_sokoban.envs:SokobanEnv",
        "max_distinct_rooms": -1,
        "reward_shaping": "dense",
        "mode": "rgb_array",
        "dim_room": (9,9),
        "num_boxes": 4,
        "game_mode": "NoAlice",
        "num_gen_steps": None,
        "max_steps": 200,
        "num_envs": 4,
        "seed": None}
    env = get_env_creator(**env_kwarg)()
    env.reset()
    #env.restore_full_state([state]*4)
    value = ValuePerfect(env, root=tuple(state))
    done = False
    optimal_path = []
    env.restore_full_state([state]*4)
    frames = [env.render()[0]]

    while not done:
        env.restore_full_state([state]*4)
        o, r, d, _ = env.step([0, 1, 2, 3])
        states = env.clone_full_state()
        _, astar = max([(r[a]+value(states[a]).item(), a) for a in range(4)])
        state = states[astar]
        done = d[astar]
        optimal_path.append(astar)
        frames.append(o[astar])

    # Create an image of all states
    print(len(optimal_path))
    img_dim = (11, 18)
    padding = img_dim[0] * img_dim[1] - len(frames) // img_dim[1] + 1
    img_pad = np.zeros_like(frames[0])
    frames += [img_pad]*padding
    from PIL import Image
    imgs = []
    for i in range(img_dim[0]):
        imgs.append(np.concatenate(frames[i*img_dim[1]:(i+1)*img_dim[1]], axis=1))
    img = np.concatenate(imgs, axis=0)
    Image.fromarray(img, "RGB").save("hard_sokoban_solution.png")

    # Create a video
    env_kwarg['num_envs'] = 1
    env_rec = get_env_creator(**env_kwarg)()
    env_rec.reset()
    env_rec = monitor.EvaluationMonitor(RestoreStateWrapper(env_rec, state0),
                                        directory='videos',
                                        force=True,
                                        video_callable=lambda *args: True
                                        )
    env_rec.reset()
    for a in optimal_path:
        env_rec.step(a)

def main():
    # test_tiny_rgb_mode()
    # test_one_hot_mode()
    # test_seed()
    #test_seed_2('rgb_array', seed=1234567)
    # test_seed_2('tiny_rgb_array', seed=1234567)
    # stress_test(seed=123)
    # test_recover()
    # test_play()
    #for _ in range(100):
    #    test_seed(dim=(10, 10), num_boxes=4)
    #for num_boxes in range(1, 5):
    #    for _ in range(100):
    #        test_type_counts(num_boxes=num_boxes)
    #test_seed(dim=(8, 8), num_boxes=2, seed=0xC1F3FBAD)
    #test_curriculum(dim=(8, 8), num_boxes=2, seed=0xC1F3FBAD, curriculum=20)
    #print(test_serialization())
    #test_curriculum_2()
    test_draw_hard_level()


if __name__ == '__main__':
    main()
