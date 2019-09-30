import os
import sys
import time

import gin.tf
import numpy as np
import tensorflow as tf

from ourlib.summary.summary_helper import SummaryHelper
from ourlib.timer import start_timer, elapsed_time_ms
from dopamine.atari.run_experiment import create_atari_environment
from dopamine.common import logger, checkpointer, iteration_statistics

PROFILING_FREQ_STEPS = 6000

@gin.configurable
class Runner(object):
    """Object that handles running Atari 2600 experiments.

  Here we use the term 'experiment' to mean simulating interactions between the
  agent and the environment and reporting some statistics pertaining to these
  interactions.

  A simple scenario to train a DQN agent is as follows:

  ```python
  base_dir = '/tmp/simple_example'
  def create_agent(sess, environment):
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
  runner = Runner(base_dir, create_agent, game_name='Pong')
  runner.run()
  ```
  """

    def __init__(self,
                 base_dir,
                 agent_creator,
                 create_environment_fn=create_atari_environment,
                 game_name=None,
                 checkpoint_file_prefix='ckpt',
                 logging_file_prefix='log',
                 log_every_n=1,
                 num_iterations=200,
                 training_steps=250000,
                 evaluation_steps=125000,
                 max_steps_per_episode=27000):
        """Initialize the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      agent_creator: A function that takes as args a Tensorflow session and an
        Atari 2600 Gym environment, and returns an agent.
      create_environment_fn: A function which receives a game name and creates
        an Atari 2600 Gym environment.
      game_name: str, name of the Atari 2600 domain to run.
      sticky_actions: bool, whether to enable sticky actions in the environment.
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      logging_file_prefix: str, prefix to use for the log files.
      log_every_n: int, the frequency for writing logs.
      num_iterations: int, the iteration number threshold (must be greater than
        start_iteration).
      training_steps: int, the number of training steps to perform.
      evaluation_steps: int, the number of evaluation steps to perform.
      max_steps_per_episode: int, maximum number of steps after which an episode
        terminates.

    This constructor will take the following actions:
    - Initialize an environment.
    - Initialize a `tf.Session`.
    - Initialize a logger.
    - Initialize an agent.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.
    """
        assert base_dir is not None
        self._logging_file_prefix = logging_file_prefix
        self._log_every_n = log_every_n
        self._num_iterations = num_iterations
        self._training_steps = training_steps
        self._evaluation_steps = evaluation_steps
        self._max_steps_per_episode = max_steps_per_episode
        self._base_dir = base_dir
        self._create_directories()
        self._summary_writer = tf.summary.FileWriter(self._base_dir)

        self._environment = create_environment_fn()
        # Set up a session and initialize variables.
        self._sess = tf.Session('',
                                config=tf.ConfigProto(allow_soft_placement=True))
        self._agent = agent_creator(self._sess, self._environment,
                                    summary_writer=self._summary_writer)
        self._summary_writer.add_graph(graph=tf.get_default_graph())
        self._sess.run(tf.global_variables_initializer())

        self._summary_helper = SummaryHelper(self._summary_writer)

        self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)
        self._steps_done = 0

        self._total_timer = None

    def _create_directories(self):
        """Create necessary sub-directories."""
        self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
        self._logger = logger.Logger(os.path.join(self._base_dir, 'logs'))

    def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
        """Reloads the latest checkpoint if it exists.

    This method will first create a `Checkpointer` object and then call
    `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
    checkpoint in self._checkpoint_dir, and what the largest file number is.
    If a valid checkpoint file is found, it will load the bundled data from this
    file and will pass it to the agent for it to reload its data.
    If the agent is able to successfully unbundle, this method will verify that
    the unbundled data contains the keys,'logs' and 'current_iteration'. It will
    then load the `Logger`'s data from the bundle, and will return the iteration
    number keyed by 'current_iteration' as one of the return values (along with
    the `Checkpointer` object).

    Args:
      checkpoint_file_prefix: str, the checkpoint file prefix.

    Returns:
      start_iteration: int, the iteration number to start the experiment from.
      experiment_checkpointer: `Checkpointer` object for the experiment.
    """
        self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                       checkpoint_file_prefix)
        self._start_iteration = 0
        # Check if checkpoint exists. Note that the existence of checkpoint 0 means
        # that we have finished iteration 0 (so we will start from iteration 1).
        latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
            self._checkpoint_dir)
        if latest_checkpoint_version >= 0:
            experiment_data = self._checkpointer.load_checkpoint(
                latest_checkpoint_version)
            if self._agent.unbundle(
                    self._checkpoint_dir, latest_checkpoint_version, experiment_data):
                assert 'logs' in experiment_data
                assert 'current_iteration' in experiment_data
                self._logger.data = experiment_data['logs']
                self._start_iteration = experiment_data['current_iteration'] + 1
                tf.logging.info('Reloaded checkpoint and will start from iteration %d',
                                self._start_iteration)

    def _initialize_episode(self):
        """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
        initial_observation = self._environment.reset()
        return self._agent._begin_episode(initial_observation)

    def _run_one_step(self, action):
        """Executes a single step in the environment.

    Args:
      action: int, the action to perform in the environment.

    Returns:
      The observation, reward, and is_terminal values returned from the
        environment.
    """
        self._steps_done += 1
        timer = start_timer()
        observation, reward, is_terminal, info = self._environment._step(action)
        env_step_elapsed_time_ms = elapsed_time_ms(timer)

        self._summary_helper.add_simple_summary('profiling/env_step_ms', env_step_elapsed_time_ms,
                                                freq=3000,
                                                global_step=self._steps_done)

        return observation, reward, is_terminal, info

    def _end_episode(self, reward):
        """Finalizes an episode run.

    Args:
      reward: float, the last reward from the environment.
    """
        self._agent._end_episode(reward)

    def _run_one_episode(self):
        """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
        step_number = 0
        total_reward = 0.

        action = self._initialize_episode()
        is_terminal = False

        # Keep interacting until we reach a terminal state.
        while True:
            observation, reward, done, info = self._run_one_step(action)
            # print(step_number, observation.shape)

            total_reward += reward
            step_number += 1

            # Perform reward clipping.
            reward = np.clip(reward, -1, 1)

            if (done or
                    step_number == self._max_steps_per_episode):
                # Stop the run loop once we reach the true end of episode.
                break
            elif info.get('is_terminal', False):
                # If we lose a life but the episode is not over, signal an artificial
                # end of episode to the agent.
                self._agent._end_episode(reward)
                action = self._agent._begin_episode(observation)
            else:

                agent_step_timer = start_timer()
                action = self._agent._step(reward, observation)

                self._summary_helper.add_simple_summary('profiling/agent_step_ms',
                                                        elapsed_time_ms(agent_step_timer),
                                                        freq=PROFILING_FREQ_STEPS,
                                                        global_step=self._steps_done)

                self._summary_helper.add_simple_summary('profiling/total_time_per_step_ms',
                                                        elapsed_time_ms(self._total_timer) / float(self._steps_done),
                                                        freq=PROFILING_FREQ_STEPS,
                                                        global_step=self._steps_done)

        self._end_episode(reward)

        return step_number, total_reward

    def _run_one_phase(self, min_steps, statistics, run_mode_str):
        """Runs the agent/environment loop until a desired number of steps.

    We follow the Machado et al., 2017 convention of running full episodes,
    and terminating once we've run a minimum number of steps.

    Args:
      min_steps: int, minimum number of steps to generate in this phase.
      statistics: `IterationStatistics` object which records the experimental
        results.
      run_mode_str: str, describes the run mode for this agent.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the sum of
        returns (float), and the number of episodes performed (int).
    """
        step_count = 0
        num_episodes = 0
        sum_returns = 0.

        while step_count < min_steps:
            episode_length, episode_return = self._run_one_episode()
            statistics.append({
                '{}_episode_lengths'.format(run_mode_str): episode_length,
                '{}_episode_returns'.format(run_mode_str): episode_return
            })
            step_count += episode_length
            sum_returns += episode_return
            num_episodes += 1
            # We use sys.stdout.write instead of tf.logging so as to flush frequently
            # without generating a line break.
            sys.stdout.write('Steps executed: {} '.format(step_count) +
                             'Episode length: {} '.format(episode_length) +
                             'Return: {}\r'.format(episode_return))
            sys.stdout.flush()
        return step_count, sum_returns, num_episodes

    def _run_train_phase(self, statistics):
        """Run training phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: The average reward generated in this phase.
    """
        # Perform the training phase, during which the agent learns.
        self._agent.eval_mode = False
        start_time = time.time()

        number_steps, sum_returns, num_episodes = self._run_one_phase(
            self._training_steps, statistics, 'train')

        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        statistics.append({'train_average_return': average_return})
        time_delta = time.time() - start_time
        tf.logging.info('Average undiscounted return per training episode: %.2f',
                        average_return)
        tf.logging.info('Average training steps per second: %.2f',
                        number_steps / time_delta)
        return num_episodes, average_return

    def _run_eval_phase(self, statistics):
        """Run evaluation phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
    """
        # Perform the evaluation phase -- no learning.
        self._agent.eval_mode = True
        _, sum_returns, num_episodes = self._run_one_phase(
            self._evaluation_steps, statistics, 'eval')
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        tf.logging.info('Average undiscounted return per evaluation episode: %.2f',
                        average_return)
        statistics.append({'eval_average_return': average_return})
        return num_episodes, average_return

    def _run_one_iteration(self, iteration):
        """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. The interleaving of train/eval phases implemented here
    are to match the implementation of (Mnih et al., 2015).

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
        statistics = iteration_statistics.IterationStatistics()
        tf.logging.info('Starting iteration %d', iteration)
        num_episodes_train, average_reward_train = self._run_train_phase(
            statistics)
        num_episodes_eval, average_reward_eval = self._run_eval_phase(
            statistics)

        self._save_tensorboard_summaries(iteration, num_episodes_train,
                                         average_reward_train, num_episodes_eval,
                                         average_reward_eval)
        return statistics.data_lists

    def _save_tensorboard_summaries(self, iteration,
                                    num_episodes_train,
                                    average_reward_train,
                                    num_episodes_eval,
                                    average_reward_eval):
        """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_train: int, number of training episodes run.
      average_reward_train: float, The average training reward.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
    """
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Train/NumEpisodes',
                             simple_value=num_episodes_train),
            tf.Summary.Value(tag='Train/AverageReturns',
                             simple_value=average_reward_train),
            tf.Summary.Value(tag='Eval/NumEpisodes',
                             simple_value=num_episodes_eval),
            tf.Summary.Value(tag='Eval/AverageReturns',
                             simple_value=average_reward_eval)
        ])
        self._summary_writer.add_summary(summary, iteration)

    def _log_experiment(self, iteration, statistics):
        """Records the results of the current iteration.

    Args:
      iteration: int, iteration number.
      statistics: `IterationStatistics` object containing statistics to log.
    """
        self._logger['iteration_{:d}'.format(iteration)] = statistics
        if iteration % self._log_every_n == 0:
            self._logger.log_to_file(self._logging_file_prefix, iteration)

    def _checkpoint_experiment(self, iteration):
        """Checkpoint experiment data.

    Args:
      iteration: int, iteration number for checkpointing.
    """
        experiment_data = self._agent.bundle_and_checkpoint(self._checkpoint_dir,
                                                            iteration)
        if experiment_data:
            experiment_data['current_iteration'] = iteration
            experiment_data['logs'] = self._logger.data
            self._checkpointer.save_checkpoint(iteration, experiment_data)

    def run_experiment(self):
        """Runs a full experiment, spread over multiple iterations."""
        tf.logging.info('Beginning training...')
        if self._num_iterations <= self._start_iteration:
            tf.logging.warning('num_iterations (%d) < start_iteration(%d)',
                               self._num_iterations, self._start_iteration)
            return

        self._total_timer = start_timer()

        for iteration in range(self._start_iteration, self._num_iterations):
            statistics = self._run_one_iteration(iteration)
            self._log_experiment(iteration, statistics)
            self._checkpoint_experiment(iteration)
