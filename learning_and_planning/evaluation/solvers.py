import typing
from itertools import count

from gym import Env

Observation = typing.Any  # TODO: specify more exact type
Action = int


class Solver:
    """ The most general class needed by evaluator.

    It's enough to implement only `solve` method, which solves one given level.
    """
    def solve(self, env: Env, budget: typing.Optional[int] = None) -> typing.Tuple[bool, int]:
        """
        Solves single environment level.

        Args:
            env: environment
            budget: if given, limits the number of steps allowed for solving the level

        Returns:
            tuple (solved_level, number_of_steps) indicating whether the level was solved
            and how many steps were used.

        NOTE: current evaluation assumes that `solve` method is honest and it doesn't cheat
        (e.g. reporting that the level was solved even if it wasn't)
        """
        raise NotImplementedError

    def stats_summary(self) -> dict:
        return {}


class ActionByActionSolver(Solver):
    """ Solver which solves level by simply choosing sequence of actions. """
    def choose_action(self, obs: Observation, env: Env) -> Action:
        raise NotImplementedError

    def solve(self, env, budget=None):
        step_ind = 0
        allowed_steps = range(1, budget + 1) if budget is not None else count(1)

        obs = env.render(mode=None)
        for step_ind in allowed_steps:
            action = self.choose_action(obs, env)
            obs, _, done, info = env.step(action)
            if done:
                if info.get('all_boxes_on_target'):
                    return True, step_ind
                else:
                    return False, step_ind

        return False, step_ind


class PolicySolver(ActionByActionSolver):
    def __init__(self, policy):
        self.policy = policy

    def choose_action(self, obs, env):
        state = env.clone_full_state()
        return self.policy.act(state, return_single_action=True)


class SamplingSolver(PolicySolver):
     def choose_action(self, obs, env):
        state = env.clone_full_state()
        return self.policy.sample(state)
