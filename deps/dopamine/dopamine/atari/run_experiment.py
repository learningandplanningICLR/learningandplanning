# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module defining classes and helper methods for running Atari 2600 agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dopamine.atari import preprocessing
import gym

import gin.tf


def load_gin_configs(gin_files, gin_bindings):
    """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
  """
    gin.parse_config_files_and_bindings(gin_files,
                                        bindings=gin_bindings,
                                        skip_unknown=False)


def create_atari_environment(game_name, sticky_actions=True):
    """Wraps an Atari 2600 Gym environment with some basic preprocessing.

  This preprocessing matches the guidelines proposed in Machado et al. (2017),
  "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open
  Problems for General Agents".

  The created environment is the Gym wrapper around the Arcade Learning
  Environment.

  The main choice available to the user is whether to use sticky actions or not.
  Sticky actions, as prescribed by Machado et al., cause actions to persist
  with some probability (0.25) when a new command is sent to the ALE. This
  can be viewed as introducing a mild form of stochasticity in the environment.
  We use them by default.

  Args:
    game_name: str, the name of the Atari 2600 domain.
    sticky_actions: bool, whether to use sticky_actions as per Machado et al.

  Returns:
    An Atari 2600 environment with some standard preprocessing.
  """

    def get_env_id(game_name, sticky_actions):
        game_version = 'v0' if sticky_actions else 'v4'
        full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
        return full_game_name

    env_id = get_env_id(game_name, sticky_actions)
    env = gym.make(env_id)
    # logger.info('env_id {}'.format(env_id))

    # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
    # handle this time limit internally instead, which lets us cap at 108k frames
    # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
    # restoring states.
    env = env.env
    env = preprocessing.AtariPreprocessing(env)
    return env


