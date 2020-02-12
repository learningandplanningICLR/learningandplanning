import numpy as np
from mpi4py import MPI
from collections import deque
import time
from ourlib.distributed_utils import TAGS
import gin.tf
import math
from baselines import logger
import os
import logging

from learning_and_planning.mcts.auto_ml import AutoMLCreator

log = logging.getLogger(__name__)


class GameStatisticsCollector:

    def __init__(self, win_ratio_ranges=(), game_length_ranges=(), game_solved_length_ranges=(),
                logs_prefix="", logs_suffix="", report_win_rate_logs=False):
        self.win_ratio_ranges = win_ratio_ranges
        self._win_ratio_running_avarages = [0] * len(self.win_ratio_ranges)
        self.game_length_ranges = game_length_ranges
        self._game_length_running_avarages = [0] * len(game_length_ranges)
        self.game_solved_length_ranges = game_solved_length_ranges
        self.game_solved_length_running_avarages = [0] * len(game_solved_length_ranges)
        self.logs_prefix = logs_prefix
        self.logs_suffix = logs_suffix
        self.report_win_rate_logs = report_win_rate_logs
        self.aux_running_averages = {}
        self.aux_ranges = {}
        self.counter = {'win_ratio': 0,
                        'game_length': 0,
                        'game_solved': 0}

    def update_running_averages_(self, running_avarages, ranges, value):
        for idx, range_ in enumerate(ranges):
            beta = 1. - 1./range_
            running_avarages[idx] = beta * running_avarages[idx] + (1. - beta) * value

    @staticmethod
    def dump_to_logger_(logger_, running_avarages, ranges, log_name_format_str, counter):
        for range, value in zip(ranges, running_avarages):
            log_name = log_name_format_str.format(range)
            beta = 1. - 1./range
            corr = 1. - beta ** counter  # EWA bias correction
            logger.record_tabular(log_name, value / corr)

    def update_aux_running_averages(self, key, value, ranges):
        if key not in self.aux_running_averages:
            self.aux_running_averages[key] = [0.] * len(ranges)
            self.aux_ranges[key] = ranges
            self.counter[key] = 0
        self.counter[key] += 1
        self.update_running_averages_(self.aux_running_averages[key], self.aux_ranges[key], value)

    def update_running_averages(self, game, is_solved):
        for key in ['win_ratio', 'game_length', 'game_solved']:
            self.counter[key] += 1
        self.update_running_averages_(self._win_ratio_running_avarages, self.win_ratio_ranges, float(is_solved))
        self.update_running_averages_(self._game_length_running_avarages, self.game_length_ranges, len(game))
        if is_solved:
            self.update_running_averages_(self.game_solved_length_running_avarages,
                                         self.game_solved_length_ranges, len(game))

    def dump_to_logger(self, logger_):
        win_rate_format_str = self.logs_prefix + "win_rate_{}" + self.logs_suffix
        self.dump_to_logger_(logger_, self._win_ratio_running_avarages, self.win_ratio_ranges, win_rate_format_str,
                             self.counter['win_ratio'])
        if self.report_win_rate_logs:
            log_win_rate_format_str = self.logs_prefix + "log 1-win_rate_{}" + self.logs_suffix
            #win_ratio_running_avarages_logs = [-math.log10(1-win_rate) for win_rate in self._win_ratio_running_avarages]
            # here we do correction before dump_to_logger, hence we pass counter=np.inf
            win_ratio_running_averages_logs = []
            for range, win_rate in zip(self.win_ratio_ranges, self._win_ratio_running_avarages) :
                beta = 1. - 1./range
                win_ratio_running_averages_logs.append(-math.log10(1-win_rate/(1. - beta ** self.counter['win_ratio'])))
            self.dump_to_logger_(logger_, win_ratio_running_averages_logs, self.win_ratio_ranges,
                                 log_win_rate_format_str, np.inf)


        game_length_format_str = self.logs_prefix + "game_length_{}" + self.logs_suffix
        self.dump_to_logger_(logger_, self._game_length_running_avarages, self.game_length_ranges,
                             game_length_format_str, self.counter['game_length'])

        game_solved_length_format_str = self.logs_prefix + "game_solved_length_{}" + self.logs_suffix
        self.dump_to_logger_(logger_, self.game_solved_length_running_avarages, self.game_solved_length_ranges,
                             game_solved_length_format_str, self.counter['game_solved'])

        for key in self.aux_running_averages:
            key_format_str = self.logs_prefix + key + "_{}" + self.logs_suffix
            self.dump_to_logger_(logger_, self.aux_running_averages[key],
                                 self.aux_ranges[key], key_format_str, self.counter[key])

def dummy_callback(*args):
    return False

@gin.configurable
class Server(object):
    def __init__(self,
                 value,
                 serializer,
                 training_steps=500000,  # number of steps for each worker
                 train_every_num_steps=4,
                 min_replay_history=20000,
                 log_every_n=50,
                 game_buffer_size=1,
                 exit_training_after_num_games=np.inf,
                 curriculum=None,
                 eval_worker_exists=False,
                 save_checkpoint_every_train_steps=np.inf,
                 checkpoint_dir=".",
                 exit_callback=None
                 ):

        self.min_replay_history = min_replay_history
        self._training_steps = training_steps
        self._train_every_num_steps = train_every_num_steps
        self._serializer = serializer
        self._value = value
        self._replay = self._value._replay

        self._game_buffer_size = game_buffer_size

        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()

        self.log_every_n = log_every_n
        self.exit_training_after_num_games = exit_training_after_num_games

        self._losses_labels = self._value.losses_labels

        self._curriculum = curriculum
        self.eval_worker_exists = eval_worker_exists
        self.save_checkpoint_every_train_steps = save_checkpoint_every_train_steps
        self.checkpoint_dir = checkpoint_dir
        self.exit_callback = dummy_callback if exit_callback is None else exit_callback
        self.auto_ml = AutoMLCreator()

    def run(self):

        # workers
        num_workers = self.size - 1
        first_mcts_worker = 2 if self.eval_worker_exists else 1
        active_workers = list(range(first_mcts_worker, self.size))
        worker_games_count = [0] * self.size

        # counters
        update_no = 1
        num_games = 0
        step_count = 0
        training_steps = 0
        number_of_removed_games = 0

        game_statistics_collector = GameStatisticsCollector(win_ratio_ranges=(100, 1000, 10000),
                                                            game_length_ranges=(100,), 
                                                            game_solved_length_ranges=(100,), report_win_rate_logs=False)
        # define storage for statistics
        avg_buff = deque(maxlen=num_workers)
        avg_skip_buff = deque(maxlen=num_workers)
        worker_step_count_buff = [0]*num_workers
        worker_total_step_count_buff = [0]*num_workers
        workers_results = [0.0]*num_workers

        # MPI status is needed to recognize the sender
        status = MPI.Status()

        # timing the results
        start = time.time()
        update_value = False

        loss, losses, gradients, lr = None, None, None, None
        num_logs = 0

        while active_workers:
            num_solved = self._replay.memory.add_count['solved']
            num_unsolved = self._replay.memory.add_count['unsolved']

            while training_steps * self._train_every_num_steps < step_count \
                    and num_solved + num_unsolved > self.min_replay_history:
                # TODO: This is somewhat strange as it make a big number of trains just after the threshold is overcome

                loss, losses, gradients, lr = self._value.train_step()

                training_steps += 1
                update_value = True

                if training_steps % self.save_checkpoint_every_train_steps == 0:
                    self._value.checkpoint(self.checkpoint_dir, training_steps)

            if update_value:
                self._value.push()

                update_value = False

            if self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAGS.GAME, status=status):
                serializer_data = np.zeros(self._serializer.buffer_size, dtype=np.uint8)
                source = status.Get_source()
                self.comm.Recv(serializer_data, source=source, tag=TAGS.GAME)

                if source not in active_workers:
                    continue

                deserialized_data = self._serializer.deserialize(np.copy(serializer_data))
                number_of_removed_games += deserialized_data['number_of_removed_games']
                worker_step_count = deserialized_data['worker_step_count']
                worker_total_step_count = deserialized_data['worker_total_step_count']
                worker_step_count_buff[source-1] = worker_step_count
                worker_total_step_count_buff[source-1] = worker_total_step_count

                for i in range(self._game_buffer_size):
                    game_steps = deserialized_data[f'game_steps_{i}']
                    if game_steps == 0:
                        continue
                    game = deserialized_data[f'game_{i}']
                    is_solved = deserialized_data[f'game_solved_{i}']
                    graph_size = deserialized_data[f'graph_size_{i}']
                    game_ensemble_std = deserialized_data[f'game_ensemble_std_{i}']
                    game = game[:game_steps]
                    worker_result = deserialized_data[f'worker_avg_result_{i}']  # TODO(pm): refactor - rename
                    avg = deserialized_data[f'avg_{i}']
                    avg_skip = deserialized_data[f'avg_skip_{i}']

                    avg_buff.append(avg)
                    avg_skip_buff.append(avg_skip)

                    num_games += 1
                    step_count += len(game)
                    worker_games_count[source] += 1
                    game_statistics_collector.update_running_averages(game, is_solved)
                    game_statistics_collector.update_aux_running_averages('graph_size', graph_size, (100,))
                    game_statistics_collector.update_aux_running_averages('game_ensemble_std', game_ensemble_std, (100,))

                    # log_names = [
                    #     f"wm_next_frame_errors_rate",
                    #     f"wm_reward_errors_rate",
                    #     f"wm_missed_done_rate",
                    #     f"wm_false_done_rate",
                    #     f"wm_any_missed_done",
                    #     f"wm_any_false_done",
                    # ]
                    # for name in log_names:
                    #     game_statistics_collector.update_aux_running_averages(
                    #         name,
                    #         deserialized_data[name + f"_{i}"],
                    #         (100,)
                    #     )

                    game_statistics_collector.update_aux_running_averages(
                        'game_ensemble_std', game_ensemble_std, (100,))

                    # source-1 due to the fact that rank=0 is the server
                    workers_results[source-1] = worker_result

                    self._replay.add(game, solved=bool(is_solved))

                    if self.auto_ml.is_auto_ml_present:
                        logs = self._value.auto_ml_train(bool(is_solved),
                                                         deserialized_data.get(f'auto_ml_parameters_{i}', None))
                        self.auto_ml.consume_logs(logs)

                if worker_step_count >= self._training_steps:
                    print(f"Worker exited. Rank:{source}")
                    active_workers.remove(source)
                    print(active_workers)

                assert self.log_every_n >= self._game_buffer_size, "Setting log_every_n<game_buffer_size " \
                                                                   "might result in inconsistent results"
                if num_games > num_logs * self.log_every_n:
                    num_logs += 1
                    game_statistics_collector.dump_to_logger(logger)
                    logger.record_tabular("Step count", step_count)
                    logger.record_tabular("Num games", num_games)
                    logger.record_tabular("Worker results mean", np.mean(workers_results))
                    logger.record_tabular("Update_no", update_no)
                    logger.record_tabular("Steps per sec", step_count / (time.time()-start))
                    logger.record_tabular("Time", (time.time()-start))
                    logger.record_tabular("number_of_removed_games", number_of_removed_games)
                    logger.record_tabular("num_solved", float(self._replay.memory.add_count['solved']))
                    logger.record_tabular("num_unsolved", float(self._replay.memory.add_count['unsolved']))

                    # avg_ = np.array(avg_buff)
                    avg_skip_ = np.array(avg_skip_buff)
                    # worker_step_count_ = np.array(worker_step_count_buff)
                    worker_total_step_count_ = np.array(worker_total_step_count_buff)

                    logger.record_tabular("avg skip mean", np.mean(avg_skip_))
                    logger.record_tabular("avg skip std", np.std(avg_skip_))
                    logger.record_tabular("worker_total_step_count mean", np.mean(worker_total_step_count_))
                    logger.record_tabular("worker_total_step_count sum per sec",
                                          np.sum(worker_total_step_count_)/(time.time()-start))
                    logger.record_tabular("worker_total_step_count std", np.std(worker_total_step_count_))
                    # We log here to maintain the same logging density in the auto_ml/non-auto_ml experiment

                    if loss is not None:
                        logger.record_tabular("loss:", loss)
                        for loss_val, loss_label in zip(losses, self._losses_labels):
                            logger.record_tabular(f"loss_{loss_label}: ", loss_val)
                        for gradient_val, loss_label in zip(gradients, self._losses_labels):
                            logger.record_tabular(f"gradient_{loss_label}: ", gradient_val)
                        logger.record_tabular("learning rate", lr)

                    if self.auto_ml.is_auto_ml_present:
                        self.auto_ml.print_logs(logger)

                    # self.params['curriculum']:
                    #     env_curriculum_ = np.array(env_curriculum_buff)
                    #     num_gen_steps_ = np.array(num_gen_steps_buff)
                    #     logger.record_tabular("avg mean", np.mean(avg_))
                    #     logger.record_tabular("avg std", np.std(avg_))
                    #     logger.record_tabular("num_gen_steps mean", np.mean(num_gen_steps_))
                    #     logger.record_tabular("num_gen_steps std", np.std(num_gen_steps_))
                    #     logger.record_tabular("env_curriculum mean", np.mean(env_curriculum_))
                    #     logger.record_tabular("env_curriculum std", np.std(env_curriculum_))

                    logger.dump_tabular()

                # if num_games % self.params.get("eval_every_num_games", 10000) == 0:
                #    self._run_eval_phase()
                #    self.eval_phase += 1

                if num_games > self.exit_training_after_num_games or self.exit_callback(game_statistics_collector):
                    exit_all()

        exit_all()

    def train(self):
        pass


def exit_all():
    print("Training finished!")
    print("Server DONE!")
    MPI.COMM_WORLD.Abort()
    exit(0)


@gin.configurable
def exit_on_threshold(game_statistics_collector, index, threshold):
    val = game_statistics_collector._win_ratio_running_avarages[index]

    return val >= threshold


@gin.configurable
class ExitOnThreshold:

    def __init__(self, index, threshold, delay):
        self.index = index
        self.threshold = threshold
        self.delay = delay
        self.condition_met = False

    def __call__(self, game_statistics_collector, *args, **kwargs):
        self.condition_met = self.condition_met or \
                             game_statistics_collector._win_ratio_running_avarages[self.index] >= self.threshold
        if self.condition_met:
            self.delay -= 1
            if self.delay < 0:
                return True

        return False
