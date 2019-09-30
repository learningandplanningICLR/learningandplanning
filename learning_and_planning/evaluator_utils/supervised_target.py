import time
from enum import Enum

from learning_and_planning.supervised_loss import custom_losses, MSEsolvable


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

  VF_AND_TYPE = "vf_and_type"
  # Note that there are some deprecated names which should be mapped as follows:
  # deprecated_names = dict(
  #   vf_positive_only="vf_solvable_only",
  #   vf_sign="state_type",
  #   best_action_ignore_finall="best_action"
  # )


def final_network_activation(target: Target):
  assert isinstance(target, Target)
  activation_map = {
    Target.VF: None,
    Target.VF_SOLVABLE_ONLY: None,
    Target.BEST_ACTION: "softmax",
    Target.STATE_TYPE: "softmax",
  }
  print("target", target)
  if target in activation_map:
    final_activation = activation_map[target]
  elif target == Target.VF_AND_TYPE:
    final_activation = {
      Target.VF_SOLVABLE_ONLY.value: activation_map[Target.VF_SOLVABLE_ONLY],
      Target.STATE_TYPE.value: activation_map[Target.STATE_TYPE],
    }
  final_activation
  return final_activation


def loss_for_target(target: Target, loss=None):
  assert isinstance(target, Target)
  default_loss_map = {
    Target.VF: "mse",
    Target.VF_SOLVABLE_ONLY: "mse",
    Target.BEST_ACTION: "categorical_crossentropy",
    Target.STATE_TYPE: "categorical_crossentropy",
  }
  if not loss:
    if target in default_loss_map:
      ret = default_loss_map[target]
    elif target == Target.VF_AND_TYPE:
      ret = {
        Target.VF_SOLVABLE_ONLY.value: MSEsolvable,
        Target.STATE_TYPE.value: default_loss_map[Target.STATE_TYPE],
      }
  else:
    assert Target != Target.VF_AND_TYPE
    if loss.startswith("custom"):
      ret = custom_losses[loss[7:]]
    else:
      ret = loss
  return ret


def net_output_size_for_target(target: Target, n_actions=None):
  assert isinstance(target, Target)
  net_output_map = {
    Target.VF: 1,
    Target.VF_SOLVABLE_ONLY: 1,
    Target.BEST_ACTION: n_actions,
    Target.STATE_TYPE: 3,
  }
  if target in net_output_map:
    net_output_size = net_output_map[target]
  elif target == Target.VF_AND_TYPE:
    net_output_size = {
      Target.VF_SOLVABLE_ONLY.value: net_output_map[Target.VF_SOLVABLE_ONLY],
      Target.STATE_TYPE.value: net_output_map[Target.STATE_TYPE],
    }
  return net_output_size