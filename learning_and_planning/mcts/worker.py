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
                 ):

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
        obs_space = planner._model.env.observation_space
        self.zero_observation = np.zeros(
            shape=obs_space.shape,
            # WARNING: Setting np.int32 cause this is value used by Circular
            # replay buffer and serializer.
            dtype=np.int32,  # obs_space.dtype
        )
        self.auto_ml_creator = AutoMLCreator()


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
                f"game_{i}": [(self.zero_observation, 0.0, 0)],
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

                ################### DEBUG
                # Scratch of debug code useful when running worker only (in debugger, without MPI)
                # deserialized_data = self._serializer.deserialize(serialized_game)
                # for i in range(self.game_buffer_size):
                #     game_steps = deserialized_data[f'game_steps_{i}']
                #     game = deserialized_data[f'game_{i}']
                #     game = game[:game_steps]
                ###################

                req = self.comm.Isend(serialized_game, dest=0, tag=TAGS.GAME)
                games_to_sent = copy.copy(games_to_sent_initial)
                games_to_sent_index = 0

        while not req.Test():
            time.sleep(1)


