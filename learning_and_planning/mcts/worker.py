import copy
import string
import time
import random

import numpy as np
from mpi4py import MPI
from learning_and_planning.experiments.helpers.client_helper import \
    inject_dict_to_gin
from ourlib.distributed_utils import TAGS
import gin.tf

from learning_and_planning.mcts.auto_ml import AutoMLCreator
from contextlib import contextmanager

@contextmanager
def gin_temp_context(params_dict):
    letters = string.ascii_lowercase
    tmp_scope_name = ''.join(random.choice(letters) for _ in range(10))
    inject_dict_to_gin(params_dict, tmp_scope_name)
    tmp_scope = gin.config_scope(tmp_scope_name)
    tmp_scope.__enter__()
    yield None
    tmp_scope.__exit__(None, None, None)


@gin.configurable
class Worker(object):
    def __init__(self,
                 value,
                 planner,
                 serializer,
                 training_steps,  # number of steps for each worker
                 curriculum,
                 game_buffer_size,
                 save_unsolved_after_num_games=np.inf,
                 debug=False,
                 ):
        """

        Args:
            debug: if run debug code for deserialization and replay buffer.
                This will hit the performance, use it only for debug purposes.
        """

        self.value = value
        self._curriculum = curriculum
        self.training_steps = training_steps
        self.game_buffer_size = game_buffer_size
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.episode_max_steps = planner.episode_max_steps
        self.planner = planner
        self._serializer = serializer
        self.save_unsolved_after_num_games = save_unsolved_after_num_games
        state_space = planner._model.env.state_space
        self.zero_state = np.zeros(
            shape=state_space.shape,
            dtype=state_space.dtype,
        )
        self.auto_ml_creator = AutoMLCreator()
        self.debug = debug
        if self.debug:
            assert self.rank == 0, "It seems that debug mode is run in regular " \
                                   "MPI pipeline, is that what you want?"

    def test_deserialization_and_replay_buffer(self, serialized_game):
        """

        This should be run only for debug purposes. Useful to debug
        serialization / replay buffer without using MPI (when running Worker
        only without Server).
        """
        print("SINGLE THREAD DEBUG MODE: deserialize game")
        deserialized_data = self._serializer.deserialize(serialized_game)
        print("SINGLE THREAD DEBUG MODE: insert game into replay buffer")
        for i in range(self.game_buffer_size):
            game_steps = deserialized_data[f'game_steps_{i}']
            is_solved = deserialized_data[f'game_solved_{i}']
            game = deserialized_data[f'game_{i}']
            game = game[:game_steps]
            self.value._replay.add(game, solved=bool(is_solved))

        print("SINGLE THREAD DEBUG MODE: sample from replay buffer")
        self.value._replay.memory.sample_transition_batch(batch_size=32)
        print("SINGLE THREAD DEBUG MODE: basic test passed")

    def run(self):

        # counters
        step_count = 0
        num_games = 1
        total_step_count = 0

        # averages
        smooth_coeff = self._curriculum.smooth_coeff
        avg, avg_skip, worker_avg_result = 0.0, 0.0, 0.0

        # placholder for MPI request
        req = None

        games_to_sent_index = 0
        games_to_sent_initial = {"number_of_removed_games": 0,
                                 "worker_step_count": 0,
                                 "worker_total_step_count": 0}
        for i in range(self.game_buffer_size):
            fake_game_kwargs = {
                f"game_{i}": [(self.zero_state, 0.0, 0)],
                f"game_steps_{i}": 0,
                f"game_solved_{i}": 0,
                f"graph_size_{i}": 0,
                f"game_ensemble_std_{i}": 0.,
                f"worker_step_count_{i}": step_count,
                f"avg_{i}": 0,
                f"num_gen_steps_{i}": 0,
                f"env_curriculum_{i}": 0,
                f"avg_skip_{i}": 0,
                f"worker_avg_result_{i}": 0,
                # f"wm_next_frame_errors_rate_{i}": 0.,
                # f"wm_reward_errors_rate_{i}": 0.,
                # f"wm_missed_done_rate_{i}": 0.,
                # f"wm_false_done_rate_{i}": 0.,
                # f"wm_any_missed_done_{i}": 0.,
                # f"wm_any_false_done_{i}": 0.,
            }
            if self.auto_ml_creator.is_auto_ml_present:
                fake_game_kwargs[f"auto_ml_parameters_{i}"] = self.auto_ml_creator.fake_data()
            games_to_sent_initial.update(fake_game_kwargs)

        games_to_sent = copy.copy(games_to_sent_initial)
        unsolved_games_buffer = []

        while step_count < self.training_steps:

            self.value.update()

            params_gin_dict = {}
            auto_ml_values = None
            if self.auto_ml_creator.is_auto_ml_present:
                auto_ml_values = self.value.auto_ml_sample()
                params_gin_dict = self.auto_ml_creator.auto_ml_dispatch_parameters(auto_ml_values)

            with gin_temp_context(params_gin_dict):
                game, game_solved, info = self.planner.run_one_episode()  # game = [(state, value, action)]
            num_games += 1
            if num_games >= self.save_unsolved_after_num_games and not game_solved:
                unsolved_games_buffer.append(game[0][0])
                if len(unsolved_games_buffer) > 100:
                    unsolved_games_buffer_np = np.stack(unsolved_games_buffer)
                    np.save(rf"unsolved_{self.rank}_{num_games}", unsolved_games_buffer_np)
                    unsolved_games_buffer = []

            # compute variance
            game_ensemble_std = np.mean([np.std(node.value_acc.get()) for node in info['nodes']])
            worker_avg_result = smooth_coeff * worker_avg_result + (1 - smooth_coeff) * game_solved

            # if_done_fp = float(info["wm_errors"]["if_done_false_positives"])
            # if_done_tp = float(info["wm_errors"]["if_done_true_positives"])
            # if_done_fn = float(info["wm_errors"]["if_done_false_negatives"])
            # if_done_tn = float(info["wm_errors"]["if_done_true_negatives"])
            # if if_done_fn == 0:
            #     missed_done_rate = 0.
            # else:
            #     missed_done_rate = if_done_fn / (if_done_fn + if_done_tp)
            #
            # if if_done_fp == 0.:
            #     false_done_rate = 0.
            # else:
            #     false_done_rate = if_done_fp / (if_done_fp + if_done_tn)
            #
            # any_missed_done = if_done_fn > 0.
            # any_false_done = if_done_fp > 0.

            games_to_sent_index_mod = games_to_sent_index % self.game_buffer_size
            game_kwargs = {
                f"game_{games_to_sent_index_mod}": game,
                f"game_steps_{games_to_sent_index_mod}": len(game),
                f"graph_size_{games_to_sent_index_mod}": info['graph_size'],
                f"game_ensemble_std_{games_to_sent_index_mod}": game_ensemble_std,
                f"game_solved_{games_to_sent_index_mod}": game_solved,
                f"worker_step_count_{games_to_sent_index_mod}": 0,
                f"avg_{games_to_sent_index_mod}": avg,
                f"num_gen_steps_{games_to_sent_index_mod}": 0,
                f"env_curriculum_{games_to_sent_index_mod}": 0,
                f"avg_skip_{games_to_sent_index_mod}": avg_skip,
                f"worker_avg_result_{games_to_sent_index_mod}": worker_avg_result,
                # f"wm_next_frame_errors_rate_{games_to_sent_index_mod}": float(info["wm_errors"]["next_frame"]) / len(game),
                # f"wm_reward_errors_rate_{games_to_sent_index_mod}": float(info["wm_errors"]["reward"]) / len(game),
                # f"wm_missed_done_rate_{games_to_sent_index_mod}": missed_done_rate,
                # f"wm_false_done_rate_{games_to_sent_index_mod}": false_done_rate,
                # f"wm_any_missed_done_{games_to_sent_index_mod}": any_missed_done,
                # f"wm_any_false_done_{games_to_sent_index_mod}": any_false_done,
            }
            if self.auto_ml_creator.is_auto_ml_present:
                game_kwargs[f"auto_ml_parameters_{games_to_sent_index_mod}"] = self.auto_ml_creator.data(auto_ml_values)
            games_to_sent.update(game_kwargs)
            games_to_sent_index = (games_to_sent_index + 1)
            total_step_count += len(game)

            if not req or req.Test():
                for i in range(self.game_buffer_size):
                    step_count += games_to_sent[f"game_steps_{i}"]

                games_to_sent['number_of_removed_games'] = max(games_to_sent_index - self.game_buffer_size, 0)
                games_to_sent['worker_step_count'] = step_count
                games_to_sent['worker_total_step_count'] = total_step_count

                serialized_game = self._serializer.serialize(**games_to_sent)

                if self.debug:
                    self.test_deserialization_and_replay_buffer(serialized_game)

                req = self.comm.Isend(serialized_game, dest=0, tag=TAGS.GAME)
                games_to_sent = copy.copy(games_to_sent_initial)
                games_to_sent_index = 0

        while not req.Test():
            time.sleep(1)


