import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import Callback

from learning_and_planning.supervised.supervised_target import Target


def mse_error_logs(y_true, y_pred):
  mse = ((y_pred - y_true) ** 2).mean()
  return {'val_shard_mse': mse}


def max_error_logs(y_true, y_pred):
  max_err = np.max(np.abs(y_pred - y_true))
  return {'val_shard_max_abs_error': max_err}


def accuracy_logs(y_true, y_pred):
  acc = (np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)).mean()
  return {"val_shard_acc": acc}


def accuracy_logs(y_true, y_pred):
  acc = (np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)).mean()
  return {"val_shard_acc": acc}


def pixel_accuracy_logs(y_true, y_pred):
  """Assumes binary prediction of each channel for each pixel"""
  assert len(y_true.shape) == 4
  assert len(y_pred.shape) == 4
  acc = (y_true == (y_pred > 0.5)).mean()
  return {"val_shard_pixel_acc": acc}


def perfect_image_prediction_logs(y_true, y_pred):
  """Assumes binary prediction of each channel for each pixel"""
  assert len(y_true.shape) == 4
  assert len(y_pred.shape) == 4
  correct_channels = (y_true == (y_pred > 0.5))
  perfect_images = correct_channels.all(axis=3).all(axis=2).all(axis=1)
  return {"val_shard_perfect_image_ratio": perfect_images.mean()}


def perfect_image_predictions_multihead_logs(y_true, y_pred):
  # Head 0 is for "next_frame" prediction.
  key = Target.NEXT_FRAME.value
  return perfect_image_prediction_logs(y_true[key], y_pred[0])


def two_head_accuracy_logs(y_true, y_pred):
  # Head 1 is for classification.
  key = Target.STATE_TYPE.value
  acc = (np.argmax(y_true[key], axis=1) == np.argmax(y_pred[1], axis=1)).mean()
  return {"val_shard_acc": acc}


def if_done_accuracy_logs(y_true, y_pred):
  # Head 1 is for "if_done" prediction.
  key = "if_done"
  acc = (np.argmax(y_true[key], axis=1) == np.argmax(y_pred[1], axis=1)).mean()
  return {"val_shard_if_done_acc": acc}


def two_head_mse_error_logs(y_true, y_pred):
  y_true = y_true[Target.VF_SOLVABLE_ONLY.value]
  # Head 0 is for value regression
  y_pred = y_pred[0]
  solvable_mask = y_true != UNSOLVABLE_FLOAT
  mse = ((y_pred[solvable_mask] - y_true[solvable_mask]) ** 2).mean()
  return {'val_shard_mse': mse}


def group_error_by_target_logs(y_true, y_pred):
  """Calculate MSE for separate groups of target values."""
  df = pd.DataFrame(dict(
    y_pred=y_pred.reshape((-1,)),
    y_true_groups=y_true.reshape((-1,)).round(0),
    y_true=y_true.reshape((-1,)),
  ))
  errs = df.groupby('y_true_groups').agg(
    lambda df_: ((df_.y_true - df_.y_pred) ** 2).mean()
  ).iloc[:, 0]  # agg returns here DataFrame with single column
  logs = dict()
  for k, v in errs.items():
    name = "z_val_shard_mse_for_target_around_{}".format(k)
    logs[name] = v
  return logs


class ValidationSetCallback(Callback):
  def __init__(self, x_val, y_val, validate_every_batch=1000,
               validation_functions=[]):
    self.x_val = x_val
    self.y_val = y_val
    self.validate_every = validate_every_batch
    self.validation_functions = validation_functions

  def on_batch_end(self, batch, logs={}):
    if batch % self.validate_every == 0 and self.validation_functions:
      y_pred = self.model.predict(self.x_val)  # .reshape((-1,))
      if isinstance(y_pred, np.ndarray):
        assert y_pred.shape == self.y_val.shape, \
          "got unequal shapes {} {}".format(y_pred.shape, self.y_val.shape)
      for vf in self.validation_functions:
        updates = vf(self.y_val, y_pred)
        # keras callbacks in tensorflow<1.13 requires values to be numpy
        # skalars
        updates = {key: np.array(value) for key, value in updates.items()}
        logs.update(updates)
