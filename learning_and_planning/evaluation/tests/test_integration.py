from gym_sokoban.envs import SokobanEnv
from learning_and_planning.evaluation.metrics import Evaluator
from learning_and_planning.evaluation.solvers import PolicySolver
from learning_and_planning.mcts.value import ValueFromOneHeadNet, PolicyFromValue
from learning_and_planning.utils import wrappers

MODEL_PATH = 'learning_and_planning/models/epoch.0340.hdf5'
ENV_KWARGS = dict(dim_room=(8, 8), num_boxes=1, mode="one_hot")
VALUE_FROM_NETWORK_CLASS = ValueFromOneHeadNet

def test_value_policy_solver_integration(
        model_path=MODEL_PATH, env_kwargs=ENV_KWARGS,
        value_from_network=VALUE_FROM_NETWORK_CLASS
):
    value_fun = value_from_network(model_path, env_kwargs)
    policy = PolicyFromValue(value_fun, env_kwargs)
    solver = PolicySolver(policy)
    evaluator = Evaluator(None, env_kwargs, num_levels=10, log_interval=1, video_directory='test_value_monitor')
    evaluator.evaluate(solver)


def test_value_policy_solver_integration_with_human_debug(
        model_path=MODEL_PATH, env_kwargs=ENV_KWARGS,
        value_from_network=VALUE_FROM_NETWORK_CLASS
):
    value_fun = value_from_network(model_path, env_kwargs)
    policy = PolicyFromValue(value_fun, env_kwargs)
    solver = PolicySolver(policy)
    env = wrappers.InfoDisplayWrapper(
        wrappers.RewardPrinter(
            SokobanEnv(**env_kwargs)
        ),
        augment_observations=False,
        min_text_area_width=300
    )
    evaluator = Evaluator(env, None, num_levels=10, log_interval=1, video_directory='test_value_monitor_debug')
    evaluator.evaluate(solver)
