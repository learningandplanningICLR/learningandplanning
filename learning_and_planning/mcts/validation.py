from copy import copy, deepcopy
from enum import Enum
import time
import os

import gin

from gym_sokoban.envs import render_utils, SokobanEnv
from learning_and_planning.common_utils.neptune_utils import render_figure
from learning_and_planning.evaluator_utils import supervised_data
from learning_and_planning.evaluator_utils.supervised_target import Target

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

shards_data_base = {
    "sokoban_8_8_1": ("/net/archive/groups/plggrl_algo/evaluator_utils/mrunner_scratch/plgkcz_sandbox/sokoban-generate_d9oczu961z/",
                      "shard_size_8_8_boxes_1_generation_v2"),
    "sokoban_8_8_2": ("/net/archive/groups/plggrl_algo/evaluator_utils/mrunner_scratch/plgkcz_sandbox/sokoban-generate_ibyc1i5oi3/",
                      "shard_size_8_8_boxes_2_generation_v2"),
    "sokoban_8_8_3": ("/net/archive/groups/plggrl_algo/evaluator_utils/mrunner_scratch/plgkcz_sandbox/sokoban-generate_tfa41ogag1/",
                      "shard_size_8_8_boxes_3_generation_v2"),
    "sokoban_8_8_4": ("/net/archive/groups/plggrl_algo/evaluator_utils/mrunner_scratch/plgkcz_sandbox/sokoban-generate_vmnd1nvm8r/",
                      "shard_size_8_8_boxes_4_generation_v2"),
    "sokoban_10_10_1": (
    "/net/archive/groups/plggrl_algo/evaluator_utils/mrunner_scratch/plgkcz_sandbox/sokoban-generate_54ugy99h6e/",
    "shard_size_10_10_boxes_1_generation_v2"),
    "sokoban_10_10_2": (
    "/net/archive/groups/plggrl_algo/evaluator_utils/mrunner_scratch/plgkcz_sandbox/sokoban-generate_r1njc5f8bq/",
    "shard_size_10_10_boxes_2_generation_v2"),
    "sokoban_10_10_4": (
    "/net/archive/groups/plggrl_algo/evaluator_utils/mrunner_scratch/plgkcz_sandbox/sokoban-generate_uerr207uwx/",
    "shard_size_10_10_boxes_4_generation_v2")
}


class CLUSTERS(Enum):
    PROMETHEUS = 0
    EAGLE = 1
    OTHER = 2


def infer_cluster():
    import socket
    host_name = socket.getfqdn()
    if "prometheus" in host_name:
        return CLUSTERS.PROMETHEUS
    if "eagle" in host_name:
        return CLUSTERS.EAGLE
    return CLUSTERS.OTHER


def get_shards_prefix(env_kwargs, number_of_shards):
    where_am_i = infer_cluster()
    if "num_boxes" in env_kwargs:
        game_id = f'sokoban_{env_kwargs["dim_room"][0]}_{env_kwargs["dim_room"][0]}_{env_kwargs["num_boxes"]}'
    else:
        game_id = None

    if game_id not in shards_data_base:
        return None
    dir_on_prometheus = shards_data_base[game_id][0]
    prefix_on_prometheus = shards_data_base[game_id][1]
    full_prefix_on_prometheus = dir_on_prometheus + prefix_on_prometheus

    if where_am_i == CLUSTERS.PROMETHEUS:
        return full_prefix_on_prometheus

    dir_to_copy = os.getcwd()
    for idx in range(number_of_shards):
        path = full_prefix_on_prometheus + "_{:0>4d}".format(idx)
        copy_command = f"scp prometheus:{path} {dir_to_copy}"
        print(f"invoking:{copy_command}")
        os.system(copy_command)

    return os.path.join(dir_to_copy, prefix_on_prometheus)


def validate_value_function(
        value, batch_size, data_files_prefixes, env_kwargs, number_of_shards,
        **load_shard_kwargs
):
    if not type(data_files_prefixes) == list:
        data_files_prefixes = [data_files_prefixes]

    data_files_prefix = None

    for data_files_prefix_ in data_files_prefixes:
        detected_shards_num_ = supervised_data.infer_number_of_shards(data_files_prefix_)
        if detected_shards_num_ >= number_of_shards:
            data_files_prefix = data_files_prefix_
            break
    print("data_files_prefix prefix", data_files_prefix)
    if data_files_prefix is None:
        print("validate_value_function(): detected less shards on disk than passed "
              "by number_of_shards. Validation will not be performed")
        return {}, {}

    def generate_shards():
        for shard in range(number_of_shards):
            (states, values, _) = supervised_data.load_shard(
                shard, target=Target.VF, data_files_prefix=data_files_prefix,
                env_kwargs=env_kwargs, **load_shard_kwargs
            )
            yield (states, np.reshape(values, newshape=-1))

    def predict(states):
        return np.concatenate([
            value.traits.distill_batch(value(states[begin:(begin + batch_size)]))
            for begin in range(0, states.shape[0], batch_size)
        ])

    start_time = time.time()

    pred_value_batches = []
    perfect_value_batches = []
    for (states, perfect_values) in generate_shards():
        pred_value_batches.append(predict(states))
        perfect_value_batches.append(perfect_values)
    (pred_values, perfect_values) = map(np.concatenate, (
        pred_value_batches, perfect_value_batches
    ))
    perfect_alive = perfect_values != -2
    pred_alive = pred_values > 0
    alive_indices = np.where(perfect_alive)
    alive_pred = pred_values[alive_indices]
    alive_perfect = perfect_values[alive_indices]
    dead_pred = pred_values[np.where(~perfect_alive)]
    alive_mse = np.mean((alive_pred - alive_perfect) ** 2)
    alive_dead_accuracy = np.mean(perfect_alive == pred_alive)
    time_delta = time.time() - start_time
    plots = {}

    try:
        plt.clf()
        ax = sns.jointplot(alive_perfect, alive_pred, kind='kde')
        plt.sca(ax.ax_marg_y)
        sns.kdeplot(dead_pred, vertical=True, color='r')
        plots['value_joint'] = render_figure(ax.fig)
        # plt.close()
    except np.linalg.LinAlgError:
        tf.logging.warning(
            'Constant value function - can\'t plot the joint distribution.'
        )

    plots['value_by_box_pos'] = visualize_value_function_by_box_position(
        value, env_kwargs
    )
    info_by_box_pos = visualize_value_function_by_box_position(
        value, env_kwargs, extraction_fn=(
            lambda vf, inputs: vf.additional_info(inputs)
        ), cmaps=value.additional_info_cmaps
    )
    if info_by_box_pos is not None:
        plots['info_by_box_pos'] = info_by_box_pos

    return ({
                'eval_time': time_delta,
                'alive_value_mse': alive_mse,
                'alive_dead_accuracy': alive_dead_accuracy,
                'alive_frac': np.mean(perfect_alive),
                'pred_alive_frac': np.mean(pred_alive),
            }, plots)


@gin.configurable
def visualize_value_function_by_box_position(
        value,
        env_kwargs,
        min_value=None,  # Unused, kept for backward compatibility.
        max_value=None,  # Unused, kept for backward compatibility.
        start_seed=0,
        grid_size=(8, 8),
        extraction_fn=None,
        cmaps=('Blues',),
):
    """Visualizes a given value function.

    May visualize more than one thing at once - to do that, provide extraction_fn,
    a function (value function, inputs) -> infos where infos is a tuple of arrays
    of shape (batch_size,). Color maps for those can be specified by the cmaps
    argument. By default, the value function's output is visualized in blue.
    """
    env_kwargs = copy(env_kwargs)
    if "seed" in env_kwargs:
        print("visualize_value_function_by_box_position(): seed passed in "
              "env_kwargs ignored, using start_seed instead")
        env_kwargs = deepcopy(env_kwargs)
        env_kwargs.pop("seed")
    env = SokobanEnv(seed=start_seed, **env_kwargs)

    if extraction_fn is None:
        # By default, visualize the value function's output.
        extraction_fn = lambda vf, inputs: (vf.traits.distill_batch(vf(inputs)),)

    def template_and_infos_for_next_board():
        (height, width) = env.dim_room
        env.reset()
        state = env.clone_full_state()
        (room, room_structure) = render_utils.get_room_state_and_structure(
            state, env.dim_room
        )
        room = render_utils.make_standalone_state(room, room_structure)
        # template is the room without the moving box and the player.
        # We operate on its binary representation because it's much simpler.
        template = render_utils.room_to_binary_map(room)
        template[:, :, 2:4] = 0  # Clear the board.
        # Put all boxes except one on their targets.
        (target_xs, target_ys) = np.where(template[:, :, 1] == 1)
        template[target_xs[1:], target_ys[1:], 2] = 1

        def inside(pos):
            (x, y) = pos
            if x < 0 or x >= width or y < 0 or y >= height:
                return False
            return template[y, x, 0] == 1

        def set_player(room, x, y):
            room = np.copy(room)
            room[y, x, 3] = 1
            return room

        def rooms_for_box_position(x, y):
            if not inside((x, y)):
                # Position outside of the room, skip.
                return np.array([])

            if template[y, x, 2] == 1:
                # Position already occupied by a box, skip.
                return np.array([])

            room = np.copy(template)
            # Put the box in the room.
            room[y, x, 2] = 1
            player_positions = filter(inside, [
                (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)
            ])

            def assert_room_valid(room):
                # Assert that there's the right number of everything.
                assert room[:, :, 2].sum() == env.num_boxes
                assert room[:, :, 3].sum() == 1
                return room

            return np.array([
                # Put the player in all possible positions around the box.
                render_utils.room_to_one_hot(render_utils.binary_map_to_room(
                    assert_room_valid(set_player(room, x, y))
                ))
                for (x, y) in player_positions
            ])

        info_grids = None
        for y in range(height):
            for x in range(width):
                rooms = rooms_for_box_position(x, y)
                if rooms.shape[0] > 0:
                    # Pick the highest possible value of player positions around the box.
                    max_index = np.argmax(value.traits.distill_batch(value(rooms)))
                    infos = extraction_fn(value, rooms)
                    if info_grids is None:
                        info_grids = tuple(np.zeros(env.dim_room) for _ in infos)
                    # Store the visualization info under the highest value index.
                    for (info_grid, info) in zip(info_grids, infos):
                        info_grid[y, x] = info[max_index]

        assert info_grids is not None, "There are no valid positions in a room."

        return (template, info_grids)

    (grid_height, grid_width) = grid_size
    templates_and_infos = [
        [template_and_infos_for_next_board() for _ in range(grid_width)]
        for _ in range(grid_height)
    ]
    num_infos = len(cmaps)
    all_infos = tuple(
        np.array([
            [infos[info_index] for (_, infos) in row]
            for row in templates_and_infos
        ])
        for info_index in range(num_infos)
    )
    info_ranges = tuple(
        # Skip zeros when computing quantiles - those are invalid positions.
        tuple(np.quantile(infos[infos != 0], q) for q in (0.05, 0.95))
        for infos in all_infos
    )

    def visualize_board_info(template, info, info_range, cmap):
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        (min_info, max_info) = info_range
        img = plt.imshow(
            info,
            interpolation='nearest',
            vmin=min_info,
            vmax=max_info,
        )
        img.set_cmap(cmap)
        plt.axis('off')

        observation = render_utils.render_state(
            render_utils.binary_map_to_room(template)
        )
        heatmap = render_figure(fig, size=observation.shape[:2])
        # Overlay the heatmap on "empty" pixels.
        mask = (observation < 64).min(axis=-1)
        observation[mask] = heatmap[mask]
        plt.close(fig)
        return observation

    def visualize_board(template, infos):
        return np.concatenate([
            visualize_board_info(template, info, info_range, cmap)
            for (info, info_range, cmap) in zip(infos, info_ranges, cmaps)
        ], axis=1)

    def delimit(image):
        (height, width, num_channels) = image.shape
        image = np.concatenate(
            (image, np.full((1, width, num_channels), 0)),
            axis=0,
        )
        return np.concatenate(
            (image, np.full((height + 1, 1, num_channels), 0)),
            axis=1,
        )

    try:
        return np.concatenate([
            np.concatenate([
                delimit(visualize_board(template, infos))
                for (template, infos) in row
            ], axis=1)
            for row in templates_and_infos
        ], axis=0).astype(np.uint8)
    except ValueError:
        # This happens where there's no info to visualize.
        return None
