import numpy as np
import tensorflow as tf
from pathlib2 import Path
from ourlib.distributed_utils import get_mpi_rank_or_0, get_mpi_comm_world
from ourlib.running_utils.misc import TeeStdout, TeeStderr
from baselines import logger
import baselines.common.tf_util as U
from learning_and_planning.experiments.helpers.client_helper import \
    get_configuration
from learning_and_planning.mcts.auto_ml import AutoMLCreator

from learning_and_planning.mcts.create_agent import create_agent

import gin.tf.external_configurables
# This is to invoke gin registration

from learning_and_planning.mcts.mpi_common import SERVER_RANK, EVALUATOR_RANK
from learning_and_planning.mcts.serialization import Serializer
from learning_and_planning.mcts.server import Server
from learning_and_planning.mcts.curriculum import Curriculum
from learning_and_planning.mcts.worker import Worker


@gin.configurable
class ExperimentMain(object):

    @property
    def my_rank_dir(self):
        return self.exp_dir_path / 'rank_{}'.format(self.my_rank)

    @property
    def my_rank(self):
        return get_mpi_rank_or_0()

    def main(self):
        self.comm = get_mpi_comm_world()
        self.rank = self.comm.Get_rank()
        self.comm.Barrier()
        self.setup()
        self.train()

    def setup(self):
        if self.rank == SERVER_RANK:
            params = get_configuration(print_diagnostics=True,
                                       inject_parameters_to_gin=True)

            experiment_id = params['experiment_id']
            self.comm.bcast([experiment_id], root=SERVER_RANK)
        else:
            data = None
            data = self.comm.bcast(data, root=SERVER_RANK)
            experiment_id = data[0] if self.rank == EVALUATOR_RANK else None
            params = get_configuration(print_diagnostics=True,
                                       inject_parameters_to_gin=True)

        exp_dir_path = "."
        self.exp_dir_path = Path(exp_dir_path)

        if self.my_rank == SERVER_RANK:
            if not self.exp_dir_path.is_dir():
                self.exp_dir_path.mkdir(parents=True)
        if not self.my_rank_dir.is_dir():
            self.my_rank_dir.mkdir(parents=True)

        self.tee_stdout = TeeStdout(str(self.my_rank_dir / 'stdout'), mode='w')
        self.tee_stderr = TeeStderr(str(self.my_rank_dir / 'stderr'), mode='w')

        logger.configure(format_strs=['stdout', 'log', 'csv', 'tensorboard'])

        self.params = params
        self.run_eval_worker = self.params.get("run_eval_worker", False)

    def train(self):

        if self.params.get("tf_seed", None) is not None:
            tf.random.set_random_seed(self.params["tf_seed"])
        if self.rank == SERVER_RANK:
            sess = U.make_session(num_cpu=self.params.get('server_tf_num_cpu', 2))
        else:
            sess = U.single_threaded_session()
        sess.__enter__()

        value, planner, env_init_kwargs = create_agent(sess)

        sess.run(tf.global_variables_initializer())
        if self.rank == SERVER_RANK:
            sess.run(value.initializers())

        value.sync()  # sync value weights to root 0

        state_shape = value._replay.memory._state_shape
        game_buffer_size = self.params['game_buffer_size']
        serializer = Serializer()
        serializer.add_variable("number_of_removed_games", shape=(1, 1), dtype=[np.int32])
        serializer.add_variable("worker_step_count", shape=(1, 1), dtype=[np.int32])
        serializer.add_variable("worker_total_step_count", shape=(1, 1), dtype=[np.int32])
        for i in range(game_buffer_size):
            serializer.add_variable(name=f"game_{i}",
                                    shape=(planner.episode_max_steps,  # game max len
                                           state_shape,  # state
                                           (1, 1),  # value
                                           (1, 1),  # action
                                           ),
                                    dtype=[np.int32, np.float32, np.uint8])
            serializer.add_variable(name=f"game_steps_{i}", shape=(1, 1), dtype=[np.int32])
            serializer.add_variable(name=f"game_solved_{i}", shape=(1, 1), dtype=[np.int32])
            serializer.add_variable(name=f"graph_size_{i}", shape=(1, 1), dtype=[np.int32])
            serializer.add_variable(name=f"game_ensemble_std_{i}", shape=(1, 1), dtype=[np.float32])
            serializer.add_variable(name=f"worker_step_count_{i}", shape=(1, 1), dtype=[np.int32])
            serializer.add_variable(name=f"avg_{i}", shape=(1, 1), dtype=[np.float32])
            serializer.add_variable(name=f"num_gen_steps_{i}", shape=(1, 1), dtype=[np.int32])
            serializer.add_variable(name=f"env_curriculum_{i}", shape=(1, 1), dtype=[np.float32])
            serializer.add_variable(name=f"avg_skip_{i}", shape=(1, 1), dtype=[np.float32])
            serializer.add_variable(name=f"worker_avg_result_{i}", shape=(1, 1), dtype=[np.float32])
            # serializer.add_variable(name=f"wm_next_frame_errors_rate_{i}", shape=(1, 1), dtype=[np.float32])
            # serializer.add_variable(name=f"wm_reward_errors_rate_{i}", shape=(1, 1), dtype=[np.float32])
            # serializer.add_variable(name=f"wm_missed_done_rate_{i}", shape=(1, 1), dtype=[np.float32])
            # serializer.add_variable(name=f"wm_false_done_rate_{i}", shape=(1, 1), dtype=[np.float32])
            # serializer.add_variable(name=f"wm_any_false_done_{i}", shape=(1, 1), dtype=[np.float32])
            # serializer.add_variable(name=f"wm_any_missed_done_{i}",shape=(1, 1), dtype=[np.float32])
            auto_ml_creator = AutoMLCreator()
            if auto_ml_creator.is_auto_ml_present:
                serializer.add_variable(f"auto_ml_parameters_{i}", shape=(1, len(auto_ml_creator.dims)), dtype=[np.int32])

        curriculum = Curriculum(enabled=self.params.get('curriculum', False),
                                model=planner._model,
                                curriculum_threshold=self.params.get('curriculum_threshold', 0.8),
                                curriculum_smooth_coeff=self.params.get('curriculum_smooth_coeff', 0.95),
                                curriculum_initial_value=self.params.get('curriculum_initial_value', 0.0),
                                curriculum_initial_length_random_walk=self.params.get('curriculum_initial_length_random_walk', 50),
                                curriculum_maximal_length_random_walk=self.params.get('curriculum_maximal_length_random_walk', 300),
                                curriculum_length_random_walk_delta=self.params.get("curriculum_length_random_walk_delta", 50),
                                curriculum_max_num_gen_steps=self.params.get('curriculum_max_num_gen_steps', 27),
                                curriculum_running_average_drop_on_increase=self.params.get('curriculum_running_average_drop_on_increase', 0.2),
                                )

        if self.rank == SERVER_RANK:
            server = Server(value=value,
                            serializer=serializer,
                            training_steps=self.params['training_steps'],
                            train_every_num_steps=self.params['train_every_num_steps'],
                            game_buffer_size=game_buffer_size,
                            exit_training_after_num_games=self.params.get("exit_trainig_after_num_games", np.inf),
                            curriculum=curriculum,
                            eval_worker_exists=self.run_eval_worker,
                            )
            server.run()
        elif self.rank == EVALUATOR_RANK and self.run_eval_worker:
            from learning_and_planning.mcts.evaluator_worker import \
                EvaluatorWorker
            eval_worker = EvaluatorWorker(value=value,
                                          env_kwargs=env_init_kwargs,
                                          planner=planner
                                          )
            eval_worker.run()
        else:
            worker = Worker(value=value,
                            training_steps=self.params['training_steps'],
                            planner=planner,
                            serializer=serializer,
                            game_buffer_size=game_buffer_size,
                            curriculum=curriculum,
                            )
            worker.run()


def main():
    experiment_main = ExperimentMain()
    return experiment_main.main()

if __name__ == '__main__':
    main()

