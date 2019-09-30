import tensorflow as tf

from dopamine.atari.runner import Runner
from dopamine.common import iteration_statistics
import gin.tf


@gin.configurable
class TrainRunner(Runner):
    """Object that handles running Atari 2600 experiments.

  The `TrainRunner` differs from the base `Runner` class in that it does not
  the evaluation phase. Checkpointing and logging for the train phase are
  preserved as before.
  """

    def __init__(self, base_dir, agent_creator):
        """Initialize the TrainRunner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      agent_creator: A function that takes as args a Tensorflow session and an
        Atari 2600 Gym environment, and returns an agent.
    """
        tf.logging.info('Creating TrainRunner ...')
        super(TrainRunner, self).__init__(
            base_dir=base_dir, agent_creator=agent_creator)
        self._agent.eval_mode = False

    def _run_one_iteration(self, iteration):
        """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. This method differs from the `_run_one_iteration` method
    in the base `Runner` class in that it only runs the train phase.

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
        statistics = iteration_statistics.IterationStatistics()
        num_episodes_train, average_reward_train = self._run_train_phase(
            statistics)

        self._save_tensorboard_summaries(iteration, num_episodes_train,
                                         average_reward_train)
        return statistics.data_lists

    def _save_tensorboard_summaries(self, iteration, num_episodes,
                                    average_reward):
        """Save statistics as tensorboard summaries."""
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Train/NumEpisodes', simple_value=num_episodes),
            tf.Summary.Value(
                tag='Train/AverageReturns', simple_value=average_reward),
        ])
        self._summary_writer.add_summary(summary, iteration)
