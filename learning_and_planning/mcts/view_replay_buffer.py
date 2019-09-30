"""Views the contents of a dumped replay buffer.

"""
import argparse

from PIL import Image, ImageDraw

from gym_sokoban.envs.render_utils import (
    get_room_state_and_structure, make_standalone_state, render_state
)
from learning_and_planning.mcts.replay_buffer.circular_replay_buffer_mcts import (
    PoloOutOfGraphReplayBuffer
)
from learning_and_planning.mcts.trace import get_font

from matplotlib import pyplot as plt
import numpy as np


REPLAY_CAPACITY = 10000


def render_state_and_value(state, value):
    board = render_state(state)
    (_, width, num_channels) = board.shape
    value_bar = Image.new('RGB', (width, 9))
    draw = ImageDraw.Draw(value_bar)
    font = get_font(size=8)
    text = 'v: {}'.format(value)
    draw.text((0, 0), text, font=font, fill=(255, 255, 255))
    delimiter = np.full((1, width, num_channels), 255)
    image = np.concatenate([board, np.asarray(value_bar), delimiter], axis=0)
    (height, _, _) = image.shape
    delimiter = np.full((height, 1, num_channels), 255)
    return np.concatenate([image, delimiter], axis=1)


def main(path, iteration, episode, start_step, dim_room, dim_grid):
    # Initialize and load the replay buffer.
    (room_height, room_width) = dim_room
    (grid_height, grid_width) = dim_grid
    batch_size = grid_height * grid_width
    replay_buffer = PoloOutOfGraphReplayBuffer(
        state_shape=(room_height * room_width * 2 + 1,),
        observation_shape=(dim_room + (7,)),
        replay_capacity=REPLAY_CAPACITY,
        batch_size=batch_size,
    )
    replay_buffer.load(path, iteration)

    # Sample transitions from the given episode.
    transitions = replay_buffer.sample_transition_batch(
        indices=([episode] * batch_size),
        transition_indices=list(range(start_step, start_step + batch_size))
    )

    # Stack sampled room states and values.
    states = np.stack([
        make_standalone_state(*get_room_state_and_structure(state, dim_room))
        for state in transitions[0]
    ], axis=0)
    values = transitions[2]
    (states, values) = tuple(
        np.reshape(x, dim_grid + x.shape[1:]) for x in (states, values)
    )

    # Make a grid of rendered states and values and concatenate it into one big
    # image.
    image = np.concatenate([
        np.concatenate([
            render_state_and_value(states[row, col], values[row, col])
            for col in range(grid_width)
        ], axis=1)
        for row in range(grid_height)
    ], axis=0)
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', type=str, help='Path to the checkpoint directory.'
    )
    parser.add_argument(
        'iteration', type=int, help='Iteration index.'
    )
    parser.add_argument(
        'episode', type=int, default=0, help='Index of the episode to view.'
    )
    parser.add_argument(
        'start_step', type=int, default=0, help='Index of the start step.'
    )
    parser.add_argument(
        '--dim_room', type=int, nargs=2, default=(8, 8), help='Room size.'
    )
    parser.add_argument(
        '--dim_grid',
        type=int,
        nargs=2,
        default=(4, 8),
        help='Size of the state grid.',
    )
    args = vars(parser.parse_args())
    main(**args)
