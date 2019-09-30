import os
import time
from contextlib import contextmanager
from itertools import product

import gin.tf
from mrunner.helpers.client_helper import inject_dict_to_gin

from learning_and_planning.common_utils.neptune_utils import render_figure
from learning_and_planning.mcts.create_agent import create_agent
from learning_and_planning.mcts.server import GameStatisticsCollector
from learning_and_planning.mcts.test_states import get_test_states
from learning_and_planning.mcts.validation import validate_value_function, get_shards_prefix
import numpy as np
from baselines import logger
import logging
import random
import string
from moviepy.editor import ImageSequenceClip
from PIL import Image, ImageDraw

log = logging.getLogger(__name__)


@contextmanager
def gin_temp_context(params_dict):
    letters = string.ascii_lowercase
    tmp_scope_name = ''.join(random.choice(letters) for _ in range(10))
    inject_dict_to_gin(params_dict, tmp_scope_name)
    tmp_scope = gin.config_scope(tmp_scope_name)
    tmp_scope.__enter__()
    yield None
    tmp_scope.__exit__(None, None, None)


class CustomPlanerRunner:

    def __init__(self, custom_params, sess, value_function_to_share_network):
        self.custom_params = custom_params
        with gin_temp_context(self.custom_params):
            _, self.planer, _ = create_agent(sess, value_function_to_share_network=value_function_to_share_network)

    def run_one_episode(self, seed):
        self.planer._model.env.seed(seed)
        with gin_temp_context(self.custom_params):
            return self.planer.run_one_episode()


def visualize_board_info(info, info_range, cmap, shape, mask=None, fmt=".0E"):
    # INFO: mostly copied from validation.py
    # visualize_board_info()
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    (min_info, max_info) = info_range
    import seaborn as sns
    sns.heatmap(info, annot=True, ax=ax, cmap=cmap,
                vmin=min_info, fmt=fmt, vmax=max_info,
                cbar=False, mask=mask,
                )
    plt.axis('off')
    heatmap = render_figure(fig, size=shape)
    plt.close(fig)
    return heatmap


@gin.configurable
class EvaluatorWorker(object):

    def __init__(self,
                 value,
                 env_kwargs,
                 planner,
                 batch_size=16,
                 eval_every_seconds=10,
                 num_games_per_eval=10,
                 eval_planners_params=None,
                 eval_planners_every=1,
                 record_games_every=1,
                 test_states_onehot_every=1,
                 validate_value_function_every=1,
                 report_win_rate_logs=False,
                 chain_env_heatmap=False,
                 toy_mr_heatmap=False,
                 toy_mr_games=0,
                 sokoban_ensemble_stats=False
                 ):
        self._value = value
        self.planner = planner
        self.batch_size = batch_size
        self.eval_every_seconds = eval_every_seconds
        self.num_games_per_eval = num_games_per_eval
        self.env_kwargs = env_kwargs
        self.report_win_rate_logs = report_win_rate_logs
        if "num_boxes" in env_kwargs:
            sokoban_dim_room = env_kwargs['dim_room']
            sokoban_num_boxes = env_kwargs['num_boxes']
            self.test_states_onehot = get_test_states(sokoban_dim_room, sokoban_num_boxes)
        else:
            self.test_states_onehot = []
        self.eval_planners_runners, self.planners_statistics_collectors = self.create_eval_planners(eval_planners_params)
        self.eval_planners_every = eval_planners_every
        self.record_games_every = record_games_every
        self.test_states_onehot_every = test_states_onehot_every
        self.validate_value_function_every = validate_value_function_every
        self.chain_env_heatmap = chain_env_heatmap
        self.toy_mr_heatmap = toy_mr_heatmap
        self.toy_mr_games = toy_mr_games
        if self.toy_mr_games > 0:
            self.room_last_visit = dict()  # room location tuple -> last step
        self.sokoban_ensemble_stats = sokoban_ensemble_stats
        self.sokoban_ensemble_stats_boards = None
        if self.sokoban_ensemble_stats:
            self.generate_sokoban_ensemble_stats()

    def generate_sokoban_ensemble_stats(self):
        def bfs(model):
            model.reset()
            root = model.state()
            visited, queue = {root}, [root]
            while queue:
                vertex = queue.pop(0)
                _, _, _, solved, states = model.neighbours(vertex)
                for solved, next_state in zip(solved, states):
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append(next_state)
                        if solved:
                            return visited

        from gym_sokoban_fast import SokobanEnvFast
        from learning_and_planning.mcts.env_model import ModelEnvPerfect

        env = SokobanEnvFast()
        model = ModelEnvPerfect(env, force_hashable=False)
        states = bfs(model)
        self.sokoban_ensemble_stats_boards = [state.one_hot for state in states]
        print(f"Found boards to evaluate:{len(self.sokoban_ensemble_stats_boards)} boards to evaluate value")

    def eval_ensembles_std_in_sokoban(self):
        obs = random.choices(self.sokoban_ensemble_stats_boards, k=10)
        values = self._value(obs=obs, states=None)
        logger.record_tabular("ensembles_mean", np.mean(np.abs(np.mean(values, axis=1))))
        logger.record_tabular("ensembles_std", np.mean(np.std(values, axis=1)))
        logger.dump_tabular()



    def create_eval_planners(self, eval_planners_params):
        sess = self._value._sess
        planers_runners = []
        planners_statistics_collectors = []
        if eval_planners_params is None:
            return planers_runners, planners_statistics_collectors
        for planner_name in eval_planners_params:
            planers_runners.append(CustomPlanerRunner(eval_planners_params[planner_name], sess, self._value))
            planners_statistics_collectors.append(GameStatisticsCollector(win_ratio_ranges=(100,),
                                                                          game_length_ranges=(100,),
                                                                          game_solved_length_ranges=(100,),
                                                                          logs_prefix=planner_name+"_",
                                                                          report_win_rate_logs=
                                                                          self.report_win_rate_logs))

        return planers_runners, planners_statistics_collectors

    def calc_children_indexes(self, info, step, env):
        if "nodes" in info:
            node = info['nodes'][step]
            children_indexes = [
                node.children[action].node.value_acc.index(parent_value=node.value_acc,
                                                           action=action)
                for action in range(env.action_space.n)
            ]
            children_indexes = np.array2string(
                np.array(children_indexes), precision=2, floatmode='fixed'
            )
        else:
            children_indexes = "  Unable to calculate"
        return children_indexes

    def record_games(self, step):
        env = self.planner._model.env
        video_dir = "agent_eval_videos"
        os.makedirs(video_dir, exist_ok=True)
        for game_id in range(self.num_games_per_eval):
            game, game_solved, episode_info = self.planner.run_one_episode()
            game_frames = []
            for i, (one_hot, value, action) in enumerate(game):
                env.restore_full_state_from_np_array_version(one_hot)
                game_frame = env.render(mode="rgb_array")
                obs = env.render(mode=env.mode)
                image = Image.fromarray(np.zeros_like(game_frame))
                draw = ImageDraw.Draw(image)
                net_value = self._value(np.expand_dims(obs, 0))
                children_indexes = self.calc_children_indexes(episode_info, i,
                                                              env)
                draw.text((0, 0),
                          f"PlannerValue:{value:.3f}\n"
                          f"Action:{action}\n"
                          f"NetValue:\n"
                          f"{np.array2string(net_value, precision=2, floatmode='fixed')}\n"
                          f"ChildrenIndexes, AFTER episode:\n"
                          f"{children_indexes}\n"
                          )
                info_pane = np.array(image)
                video_frame = np.concatenate((game_frame, info_pane), axis=1)
                game_frames.append(video_frame)
            for ext in ["avi", "mp4", "mpeg"]:
                video_file = os.path.join(video_dir,
                                          "{:0>5d}_{}_{:0>4d}.{}".format(step, "solved" if game_solved else "unsolved",
                                                                         game_id, ext))
                try:
                    clip = ImageSequenceClip(game_frames, fps=2)
                    clip.write_videofile(video_file)  # If failed consider adding plgrid/tools/ffmpeg/3.0 on prometheus
                    break
                except:
                    pass

    def run(self):
        step = 0
        number_of_shards = 1
        validation_set_prefix = get_shards_prefix(env_kwargs=self.env_kwargs,
                                                  number_of_shards=number_of_shards)
        log.info('validation_set_prefix:', validation_set_prefix)

        while True:
            step += 1
            self._value.update()
            if self.sokoban_ensemble_stats:
                self.eval_ensembles_std_in_sokoban()

            if step % self.eval_planners_every == 0:
                # Each planner will play the same level
                seed = np.random.randint(1 << 63)
                for planner_runner, planner_statistics_collector in zip(self.eval_planners_runners,
                                                                 self.planners_statistics_collectors):
                    game, game_solved, info = planner_runner.run_one_episode(seed)

                    for key, value in info.items():
                        if key != 'nodes':
                            planner_statistics_collector.update_aux_running_averages(key, value, (100,))

                    planner_statistics_collector.update_running_averages(game, game_solved)
                    planner_statistics_collector.dump_to_logger(logger)

                logger.dump_tabular()
            if step % self.record_games_every == 0:
                self.record_games(step)

            if self.test_states_onehot and step % self.test_states_onehot_every == 0:
                value_of_test_states = self._value(self.test_states_onehot)
                for idx, tsoh in enumerate(self.test_states_onehot):
                    logger.record_tabular("value of test state {}: ".format(idx),
                                          np.mean(value_of_test_states[idx]))

            if validation_set_prefix is not None and self.validate_value_function_every is not None \
                    and step % self.validate_value_function_every == 0:
                metrics, images = validate_value_function(
                    value=self._value,
                    batch_size=self.batch_size,
                    data_files_prefixes=validation_set_prefix,
                    env_kwargs=self.env_kwargs,
                    number_of_shards=number_of_shards,
                    sample_data=True,
                    max_samples_per_board=800,
                    seed=0,
                )
                for i, (name, image) in enumerate(images.items()):
                    image_pil = Image.fromarray(image)
                    logger.record_tabular("eval_" + name, image_pil)
                for (name, value) in metrics.items():
                    logger.record_tabular("eval_" + name, value)
                logger.dump_tabular()

            if self.chain_env_heatmap:
                self.log_chain_env_heatmaps()

            if self.toy_mr_heatmap:
                self.log_toy_mr_heatmaps()

            if self.toy_mr_games > 0:
                self.run_toy_mr_games(step)

            time.sleep(self.eval_every_seconds)

    def run_toy_mr_games(self, step):

        env = self.planner._model.env
        env.reset()
        num_visited = list()
        for game_id in range(self.toy_mr_games):
            game, game_solved, episode_info = self.planner.run_one_episode()
            visited_rooms = set()
            for i, (one_hot, value, action) in enumerate(game):
                env.restore_full_state_from_np_array_version(one_hot)
                visited_rooms.add(env.room.loc)
            for coord in visited_rooms:
                self.room_last_visit[coord] = step
            num_visited.append(len(visited_rooms))
        last_visit = np.array(list(self.room_last_visit))
        logger.record_tabular("rooms visited", last_visit.size)
        logger.record_tabular("avg room unvisited time",
                              step - np.mean(last_visit))
        logger.record_tabular("max room unvisited time",
                              np.max(step - last_visit))
        logger.record_tabular("avg visited rooms per episode",
                              np.mean(num_visited))

    def log_chain_env_heatmaps(self):
        state_size = self.env_kwargs["N"]
        mean_img = np.zeros((state_size, state_size))
        std_img = np.zeros((state_size, state_size))
        # statistics calculated over the reachable states
        # (lower left triangle)
        std_max = 0.
        std_sum = 0.
        value_sum = 0.

        for i, j in product(range(state_size), range(state_size)):
            state = np.zeros((state_size, state_size, 1))
            state[i, j] = 1.
            # this will not work properly if _value returns policy
            ensemble_values = self._value(
                np.expand_dims(state, 0)  # expand "batch" dimension
            )
            ensemble_mean = np.mean(ensemble_values)
            ensemble_std = np.std(ensemble_values)
            mean_img[i, j] = ensemble_mean
            std_img[i, j] = ensemble_std
            # update statistics for lower-left states
            if i - j >= 0:
                std_sum += ensemble_std
                std_max = max(std_max, ensemble_std)
                value_sum += ensemble_mean

        for name, stats_img, cmap, info_range in [
            ("std", std_img, 'Blues', (0, 0.1)),
            ("mean", mean_img, 'seismic', (-0.2, 0.2))
        ]:
            img_array = visualize_board_info(
                stats_img, info_range, cmap=cmap,
                shape=(state_size * 50, state_size * 50)
            )
            image_pil = Image.fromarray(img_array)
            logger.record_tabular("eval_" + name + "_img", image_pil)
        logger.record_tabular("eval_ensemble_std_sum", std_sum)
        logger.record_tabular("eval_ensemble_std_max", std_max)
        logger.record_tabular("eval_value_sum", value_sum)
        logger.dump_tabular()

    def log_toy_mr_heatmaps(self):
        # A lot of code copied from chain_env heatmaps
        env = self.planner._model.env
        env.reset()
        from toy_mr import ToyMR
        assert isinstance(env, ToyMR)
        start_state = env.clone_full_state()
        room_shape = env.starting_room.size
        keys = [
            key_coord for room_coord, key_coord in env.keys.keys()
            if room_coord == env.starting_room.loc
        ]
        doors = [
            door_coord for room_coord, door_coord in env.keys.keys()
            if room_coord == env.starting_room.loc
        ]

        std_max = 0.
        std_sum = 0.
        value_sum = 0.

        mean_imgs = dict(key_taken=None, key_untaken=None)
        std_imgs = dict(key_taken=None, key_untaken=None)
        for state_type in ["key_taken", "key_untaken"]:
            mean_img = np.zeros(room_shape)
            std_img = np.zeros(room_shape)
            for i, j in product(range(room_shape[0]), range(room_shape[1])):
                env.restore_full_state(start_state)
                if env.starting_room.map[i, j] != 0:
                    # traps or walls
                    continue
                # HACK: change agent position and key status,
                # render observation
                env.agent = (i, j)
                if state_type == "key_taken":
                    for coord, untaken in env.keys.items():
                        assert untaken
                        if coord[0] == env.starting_room.loc:
                            env.keys[coord] = False
                            env.num_keys += 1

                obs = env.render(mode=env.mode)
                # this will not work properly if _value returns policy
                ensemble_values = self._value(
                    np.expand_dims(obs, 0)  # expand "batch" dimension
                )
                ensemble_mean = np.mean(ensemble_values)
                ensemble_std = np.std(ensemble_values)
                mean_img[i, j] = ensemble_mean
                std_img[i, j] = ensemble_std
                if ((i, j) not in keys) and ((i, j) not in doors):
                    # do not include unreachable states
                    std_sum += ensemble_std
                    std_max = max(std_max, ensemble_std)
                    value_sum += ensemble_mean
            mean_imgs[state_type] = mean_img
            std_imgs[state_type] = std_img
        mean_img = np.concatenate([
            mean_imgs["key_untaken"], mean_imgs["key_taken"]
        ], axis=1)
        std_img = np.concatenate([
            std_imgs["key_untaken"], std_imgs["key_taken"]
        ], axis=1)

        room_mask = (env.starting_room.map != 0)
        heat_map_mask = np.concatenate([
            room_mask, room_mask
        ], axis=1)

        for name, stats_img, cmap, info_range in [
            ("std", std_img, 'Blues', (0, 0.1)),
            ("mean", mean_img, 'coolwarm', (-1., 1.))
        ]:
            img_array = visualize_board_info(
                stats_img.transpose(),  # toy_mr coordinates are transposed
                info_range, cmap=cmap, fmt=".1E",
                shape=(room_shape[1] * 100, room_shape[0] * 100 * 2),
                # transpose
                mask=heat_map_mask.transpose()  # transpose
            )
            image_pil = Image.fromarray(img_array)
            logger.record_tabular("eval_" + name + "_img", image_pil)
        logger.record_tabular("eval_ensemble_std_sum", std_sum)
        logger.record_tabular("eval_ensemble_std_max", std_max)
        logger.record_tabular("eval_value_sum", value_sum)
        logger.dump_tabular()