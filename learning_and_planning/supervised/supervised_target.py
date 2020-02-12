from typing import Optional

import numpy as np

from enum import Enum

from learning_and_planning.supervised.supervised_loss import custom_losses, MSEsolvable



class Target(Enum):
  # Regress perfect value function.
  VF = "vf"
  # Regress perfect value for solvable states.
  VF_SOLVABLE_ONLY = "vf_solvable_only"
  # Dead states vs solved vs solvable classification.
  STATE_TYPE = "state_type"
  # Predict best action only for solvable states, according to perfect value.
  # In case of draw lowest action is treated as target.
  BEST_ACTION = "best_action"
  # Predict state type, regress perfect value for solvable states
  VF_AND_TYPE = "vf_and_type"
  # Next frame based on current frame and action (note that different frame
  # encodings might be used.
  NEXT_FRAME = "next_frame"
  # Predict difference in value between two not to distant states.
  DELTA_VALUE = "delta_value"
  # Regress perfect discounted (gamma=0.99) value function.
  VF_DISCOUNTED = "vf_discounted"
  # As BEST_ACTION, but based on two consecutive frames of perfect solution.
  BEST_ACTION_FRAMESTACK = "best_action_framestack"
  #
  NEXT_FRAME_AND_DONE = "next_frame_and_done"


def final_network_activation(target: Target):
  assert isinstance(target, Target)
  activation_map = {
    Target.VF: None,
    Target.VF_SOLVABLE_ONLY: None,
    Target.BEST_ACTION: "softmax",
    Target.STATE_TYPE: "softmax",
    Target.NEXT_FRAME: "sigmoid",
    Target.DELTA_VALUE: None,
    Target.VF_DISCOUNTED: None,
    Target.BEST_ACTION_FRAMESTACK: "softmax",
  }
  print("target", target)
  if target in activation_map:
    final_activation = activation_map[target]
  elif target == Target.VF_AND_TYPE:
    final_activation = {
      Target.VF_SOLVABLE_ONLY.value: activation_map[Target.VF_SOLVABLE_ONLY],
      Target.STATE_TYPE.value: activation_map[Target.STATE_TYPE],
    }
  elif target == Target.NEXT_FRAME_AND_DONE:
    final_activation = {
      Target.NEXT_FRAME.value: activation_map[Target.NEXT_FRAME],
      "if_done": "sigmoid",
    }
  return final_activation


def loss_for_target(target: Target, loss=None):
  assert isinstance(target, Target)
  default_loss_map = {
    Target.VF: "mse",
    Target.VF_SOLVABLE_ONLY: "mse",
    Target.BEST_ACTION: "categorical_crossentropy",
    Target.STATE_TYPE: "categorical_crossentropy",
    Target.NEXT_FRAME: "binary_crossentropy",
    Target.DELTA_VALUE: "mse",
    Target.VF_DISCOUNTED: "mse",
    Target.BEST_ACTION_FRAMESTACK: "categorical_crossentropy",
  }
  if not loss:
    if target in default_loss_map:
      ret = default_loss_map[target]
    elif target == Target.VF_AND_TYPE:
      ret = {
        Target.VF_SOLVABLE_ONLY.value: MSEsolvable,
        Target.STATE_TYPE.value: default_loss_map[Target.STATE_TYPE],
      }
    elif target == Target.NEXT_FRAME_AND_DONE:
      ret = {
        Target.NEXT_FRAME.value: default_loss_map[Target.NEXT_FRAME],
        "if_done": "binary_crossentropy",
      }
  else:
    assert target not in [Target.VF_AND_TYPE, Target.NEXT_FRAME_AND_DONE]
    if loss.startswith("custom"):
      ret = custom_losses[loss[7:]]
    else:
      ret = loss
  return ret


def net_output_size_for_target(target: Target, n_actions=None, frame_channels=None):
  assert isinstance(target, Target)
  net_output_map = {
    Target.VF: 1,
    Target.VF_SOLVABLE_ONLY: 1,
    Target.BEST_ACTION: n_actions,
    Target.STATE_TYPE: 3,
    Target.NEXT_FRAME: frame_channels,
    Target.DELTA_VALUE: 1,
    Target.VF_DISCOUNTED: 1,
    Target.BEST_ACTION_FRAMESTACK: n_actions,
  }
  if target in net_output_map:
    net_output_size = net_output_map[target]
  elif target == Target.VF_AND_TYPE:
    net_output_size = {
      Target.VF_SOLVABLE_ONLY.value: net_output_map[Target.VF_SOLVABLE_ONLY],
      Target.STATE_TYPE.value: net_output_map[Target.STATE_TYPE],
    }
  elif target == Target.NEXT_FRAME_AND_DONE:
    net_output_size = {
      Target.NEXT_FRAME.value: net_output_map[Target.NEXT_FRAME],
      "if_done": 1,
    }
  return net_output_size


def concat_observations_for_delta_value(base_ob, close_ob):
  """Standard way of creating input for delta_value network."""
  return np.concatenate([base_ob, close_ob], axis=2)


def concat_observations_for_frame_stack(first_frame: Optional[np.ndarray],
                                        second_frame: np.ndarray):
  """Standard way of creating input for best_action_framestack network."""
  if first_frame is None:
    first_frame = np.zeros(shape=second_frame.shape, dtype=second_frame.dtype)
  return np.concatenate([first_frame, second_frame], axis=2)
