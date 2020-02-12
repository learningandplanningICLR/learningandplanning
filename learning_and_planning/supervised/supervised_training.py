import os
from collections import Counter
import numpy as np

from mrunner.helpers.client_helper import get_configuration
from gym_sokoban.envs import SokobanEnv
from learning_and_planning.supervised.supervised_data import (
    load_shard, concat_arrays_by_name, infer_number_of_shards,
    n_channels_from_mode
)
from learning_and_planning.supervised.supervised_target import Target, final_network_activation, \
  loss_for_target, net_output_size_for_target
from learning_and_planning.supervised.supervised_validation import ValidationSetCallback, \
  max_error_logs, \
  group_error_by_target_logs, mse_error_logs, accuracy_logs, \
  two_head_accuracy_logs, two_head_mse_error_logs, \
  perfect_image_prediction_logs, pixel_accuracy_logs, \
  perfect_image_predictions_multihead_logs, if_done_accuracy_logs
from learning_and_planning.nets_factory import get_network

from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import LearningRateScheduler, \
  ModelCheckpoint

from joblib import Parallel, delayed


def count_cpu():
  try:
    import multiprocessing
    count = multiprocessing.cpu_count()
  except:
    import os
    count = os.cpu_count()
  return count


class SupervisedExperiment:

  def __init__(self, *, data_files_prefix, env, net,
               epochs,
               batch_size,
               lr,
               lr_decay=0.0,
               shards_to_use=None,
               validation_shards=1,
               save_every=None,
               output_dir,
               histogram_freq=None,
               validate_every_batch=5000,
               neptune_first_batch=10000,
               target="vf",
               loss=None,
               n_cores=None,
               sample_data=False,
               max_samples_per_board=1000,
               eval_games_to_play=10,
               **kwargs):
    if shards_to_use is None:
      self.number_of_shards = infer_number_of_shards(data_files_prefix)
    else:
      self.number_of_shards = shards_to_use
    self.validation_shards = validation_shards
    assert self.validation_shards < self.number_of_shards
    if self.number_of_shards == 1:
      print("WARNING: there is only one shard, so it is used for both training "
            "and validation.")
      self.training_shards = [0]
      self.validation_shards = [0]
    else:
      self.training_shards = list(
        range(self.number_of_shards - self.validation_shards)
      )
      self.validation_shards = list(
        range(self.number_of_shards - self.validation_shards,
              self.number_of_shards)
      )

    self.data_files_prefix = data_files_prefix
    self.save_every = save_every
    self.checkpoint_dir = os.path.join(output_dir, "checkpoints",
                                       "epoch.{epoch:04d}.hdf5")
    os.makedirs(self.checkpoint_dir, exist_ok=True)
    self.exp_dir_path = output_dir
    self.histogram_freq = histogram_freq
    self.epochs = epochs
    self.env_kwargs = env
    self.render_env = SokobanEnv(**self.env_kwargs)
    self.render_mode = self.render_env.mode
    self.target = Target(target)
    del target
    print("self.target", self.target)
    self.loss = loss_for_target(self.target, loss)
    final_activation = final_network_activation(self.target)
    net_output_size = net_output_size_for_target(
        self.target, self.render_env.action_space.n,
        n_channels_from_mode(env.get("mode", "one_hot"))
    )
    input_channels = n_channels_from_mode(env.get("mode", "one_hot"))
    if self.target in [Target.NEXT_FRAME, Target.NEXT_FRAME_AND_DONE]:
      input_channels += SokobanEnv(**env).action_space.n
    if self.target in [Target.DELTA_VALUE, Target.BEST_ACTION_FRAMESTACK]:
      input_channels *= 2
    self.metrics = [self.loss]
    if isinstance(self.loss, dict):
      # [0] is a dirty change of metrics for vf_and_type
      self.metrics = self.metrics[0]
    self.network = get_network(
      input_shape=tuple(list(env["dim_room"]) + list((input_channels,))),
      output_size=net_output_size,
      final_activation=final_activation,
      **net
    )
    self.network.compile(optimizer="adam", loss=self.loss, metrics=self.metrics)
    self.learning_rate_lambda = lambda epoch: lr/(1+lr_decay*epoch)
    self.batch_size = batch_size
    self.validate_every_batch = validate_every_batch
    self.neptune_first_batch = neptune_first_batch
    if n_cores is None:
      n_cores = count_cpu()
    self.n_cores = n_cores
    self.sample_data = sample_data
    self.max_samples_per_board = max_samples_per_board
    self.random_state = np.random.RandomState(0)
    self.eval_games_to_play = eval_games_to_play

  def get_callbacks(self, valid_x, valid_y):
    callbacks = list()
    if self.target in [Target.VF, Target.VF_SOLVABLE_ONLY, Target.DELTA_VALUE, Target.VF_DISCOUNTED]:
      validation_functions = [group_error_by_target_logs, mse_error_logs]
    elif self.target in [Target.BEST_ACTION, Target.STATE_TYPE, Target.BEST_ACTION_FRAMESTACK]:
      validation_functions = [accuracy_logs,]
    elif self.target == Target.VF_AND_TYPE:
      validation_functions = [two_head_accuracy_logs, two_head_mse_error_logs]
    elif self.target == Target.NEXT_FRAME:
      validation_functions = [perfect_image_prediction_logs,
                              pixel_accuracy_logs]
    elif self.target == Target.NEXT_FRAME_AND_DONE:
      validation_functions = [perfect_image_predictions_multihead_logs,
                              if_done_accuracy_logs]
    else:
      raise ValueError("Unknow target {}".format(self.target))

    if self.target in [Target.VF, Target.VF_DISCOUNTED, Target.BEST_ACTION, Target.BEST_ACTION_FRAMESTACK]:
      agent_video_dir = os.path.join(self.exp_dir_path,
                                     "agent_evaluation_videos", )
      os.makedirs(agent_video_dir, exist_ok=True)
      callbacks.append(
        EvaluateAgentCallback(env_kwargs=self.env_kwargs,
                              agent_video_dir=agent_video_dir,
                              validate_every_batch=self.validate_every_batch,
                              games_to_play=self.eval_games_to_play,
                              target=self.target)
      )

    callbacks.extend([
      LearningRateScheduler(self.learning_rate_lambda, verbose=0),
      ValidationSetCallback(
          valid_x, valid_y, self.validate_every_batch, validation_functions),
    ])

    if self.save_every:
      callbacks.append(ModelCheckpoint(self.checkpoint_dir, verbose=100,
                                       period=self.save_every))

    if self.histogram_freq:
      callbacks.append(TensorBoard(
          write_graph=False, log_dir=os.path.join(self.exp_dir_path, "tensorboard"),
          write_grads=True, histogram_freq=self.histogram_freq))

    return callbacks

  def load_shards(self, shards):
    """

    Args:
      shards: list of integers
    """
    seeds = self.random_state.randint(2**31, size=len(shards))
    seeds = [int(seed) for seed in seeds]
    arrays = Parallel(n_jobs=self.n_cores, verbose=11)(delayed(load_shard)(
        shard, self.target, data_files_prefix=self.data_files_prefix,
        env_kwargs=self.env_kwargs, sample_data=self.sample_data,
        max_samples_per_board=self.max_samples_per_board, seed=seed,
    ) for shard, seed in zip(shards, seeds))
    x_arrays, y_arrays, _ = zip(*arrays)

    x_array = np.concatenate(x_arrays)
    y_array = concat_arrays_by_name(y_arrays)

    return x_array, y_array

  def print_array(self, array, name, indent=0, print_top=10):
    megabytes = array.nbytes / 1000000
    indent_str = " " * indent
    print(indent_str + "{} size {:.1f} M, shape {}, min {}, max {}".format(
      name, megabytes, array.shape, array.min(), array.max()))
    if np.isnan(array).any():
      print(indent_str + "WARNING: array has {:.1f}% of nan values".format(
        np.isnan(array).mean() * 100))
    if print_top:
      if array[0].size < 20:
        count = Counter(tuple(record.flatten()) for record in array)
        print(indent_str + "Top records frequency")
        for record, counts in count.most_common(print_top):
          print(indent_str + "  {}: {}".format(record[:10], counts))
      else:
        print(indent_str + "Records are too large, not counting their values "
                           "to not risk OOM error.")

  def print_array_stats(self, header, **arrays):
    print(header)
    for name, array in arrays.items():
      if isinstance(array, np.ndarray):
        self.print_array(array, name, indent=2)
      else:
        for key, arr in array.items():
          self.print_array(arr, name + " " + key, indent=2)

  def load_train_valid_data(self):
    valid_x, valid_y, = self.load_shards(self.validation_shards)
    train_x, train_y = self.load_shards(self.training_shards)
    self.print_array_stats("Train Arrays:", x=train_x, y=train_y)
    self.print_array_stats("Validation Arrays:", x=valid_x, y=valid_y)

    return dict(
        train=dict(x=train_x, y=train_y),
        valid=dict(x=valid_x, y=valid_y)
    )

  def run_experiment(self):
    data = self.load_train_valid_data()
    train_x, train_y = data['train']['x'], data['train']['y']
    valid_x, valid_y = data['valid']['x'], data['valid']['y']
    self.network.fit(
        train_x, train_y, epochs=self.epochs,
        initial_epoch=0, batch_size=self.batch_size,
        validation_data=(valid_x, valid_y),
        shuffle=True, verbose=1,
        callbacks=self.get_callbacks(valid_x, valid_y)
    )


def main():
  config = get_configuration(
    print_diagnostics=True, with_neptune=True,
    nesting_prefixes=("net_", "env_")
  )
  # You can set locally e.g. POLO_SUPERVISED_OUTPUT=/tmp/supervised_output/
  output_dir = os.environ.get("POLO_SUPERVISED_OUTPUT", os.getcwd())
  print("Output directory: {}".format(output_dir))
  experiment = SupervisedExperiment(output_dir=output_dir, **config)
  experiment.run_experiment()


if __name__ == '__main__':
  main()
