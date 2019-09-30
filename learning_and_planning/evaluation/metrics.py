import os
import tempfile
from typing import Dict, Optional

import gym
import numpy as np

from baselines import logger
from gym_sokoban.envs import SokobanEnv

from learning_and_planning.evaluation.monitor import EvaluationMonitor
from learning_and_planning.evaluation.solvers import Solver


class Evaluator:
    def __init__(
            self,
            env: Optional[gym.Env] = None,
            env_kwargs: Optional[Dict] = None,
            num_levels: int = 100,
            total_budget: Optional[int] = None,
            log_interval: int = 10,
            video_directory: Optional[str] = None,
            level_budget: Optional[int] = 1000
    ):
        """

        Arguments:
            env: SokobanEnv instance (possibly wrapped) to use.
            env_kwargs: kwargs to create new SokobanEnv. If `env` is not None, it is not used.
            num_levels: how many levels to run.
            total_budget: if not None, evaluation ends after that number of steps.
            log_interval: how often log solving stats
            video_directory: if not None, video will be recorded to given directory
                (or newly created directory if it already exists)
            level_budget: if environment max_steps attribute is set to `inf`,
                it will be overriden by this value (if not None)
        """
        self._env = env
        self.env_kwargs = env_kwargs or {}
        self.num_levels = num_levels
        self.total_budget = total_budget
        self.log_interval = log_interval
        self.video_directory = video_directory
        self.enable_video = video_directory is not None
        self.level_budget = level_budget

    @property
    def env(self):
        if self._env:
            return self._env
        return SokobanEnv(**self.env_kwargs)

    def evaluate(self, solver: Solver) -> float:
        """
        Evaluate given solver by running it predefined number of times.

        Returns:
            Ratio of solved levels.
        """
        env = self._prepare_env()
        logger.log('Evaluation on {} levels started...'.format(self.num_levels))
        solved_num = 0
        budget_used = 0
        budget_left = self.total_budget
        level_ind = 1
        for level_ind in range(1, self.num_levels + 1):
            env.reset()
            level_budget = self.level_budget or budget_left
            solved, num_steps = solver.solve(env, level_budget)
            solved_num += 1 if solved else 0
            budget_used += num_steps
            logger.debug('Level {}{} solved, budget used so far: {}'.format(level_ind, "" if solved else " not", budget_used))
            if level_ind % self.log_interval == 0:
                logger.log(
                    '    Partial update: ' +
                    'Solved: {}/{}. '.format(solved_num, level_ind) +
                    'Budget used so far: {}'.format(budget_used))

            if self.total_budget is not None:
                budget_left -= num_steps
                if budget_used >= self.total_budget:
                    logger.debug('Budget ended after {} levels'.format(level_ind))
                    break

        solved_ratio = solved_num / level_ind
        logger.log('Solved: {}/{} ({}%)'.format(solved_num, self.num_levels, solved_ratio * 100))
        if self.total_budget:
            logger.log('Budget used: {}/{}'.format(budget_used, self.total_budget))
        env.close()
        logger.log('Evaluation finished')
        return solved_ratio

    def _prepare_env(self):
        env = self.env
        if self.enable_video:
            env = self._record(env)
        if env.unwrapped.max_steps == np.inf:
            logger.info('Environment max steps was set to INF. Solver can never reach the solution.')
            if self.level_budget:
                env.unwrapped.max_steps = self.level_budget
            logger.info('Environment max steps set to {}.'.format(self.level_budget))
        return env

    def _record(self, env):
        try:
            env = EvaluationMonitor(env, directory=self.video_directory, write_upon_reset=True, video_callable=lambda v: True)
        except gym.error.Error:
            basedir = os.path.dirname(self.video_directory)
            new_dir = tempfile.mkdtemp(prefix=self.video_directory + '_', dir=basedir)
            logger.info('Directory {} exists. Created new directory: {}'.format(self.video_directory, new_dir))
            env = EvaluationMonitor(env, directory=new_dir, write_upon_reset=True, video_callable=lambda v: True)
        return env
