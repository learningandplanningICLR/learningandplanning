import gin

from learning_and_planning.envs.environments import sokoban_with_finite_number_of_games
from learning_and_planning.envs.sokoban_env_creator import get_env_creator
from learning_and_planning.mcts.env_model import ModelEnvPerfect, SimulatedSokobanEnvModel
from learning_and_planning.mcts.mcts_planner import MCTS
from learning_and_planning.mcts.mcts_planner_with_voting import MCTSWithVoting
from learning_and_planning.mcts.value import ValuePerfect
from learning_and_planning.mcts.value_trainable import ValueVanilla, ValueEnsemble, ValueZero, MPIValueWrapper
from learning_and_planning.mcts.replay_buffer import circular_replay_buffer_mcts

# Import to register models
# noinspection PyUnresolvedReferences
import learning_and_planning.mcts.models
# noinspection PyUnresolvedReferences
# import learning_and_planning.nets_factory

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


@gin.configurable
def use_perfect_env(value=True):
    return value

# noinspection PyArgumentList
@gin.configurable
def create_agent(sess,
                 agent_name,
                 value_function_name,
                 replay_capacity,
                 use_staging=True,
                 value_function_to_share_network=None):

    env = get_env_creator(num_envs=1)()
    env.reset()
    if use_perfect_env():
        true_model = ModelEnvPerfect(env, force_hashable=False)
        model = ModelEnvPerfect(env, force_hashable=False)
    else:
        true_model = ModelEnvPerfect(env, force_hashable=False)
        model = SimulatedSokobanEnvModel(env, force_hashable=False)

    state_shape, state_dtype = model.state_space()
    obs_shape, obs_dtype = model.observation_space()
    renderer = model.renderer()

    # creates replay buffer
    replay = circular_replay_buffer_mcts.PoloWrappedReplayBuffer(
        replay_capacity=replay_capacity,
        state_shape=state_shape,
        observation_shape=obs_shape,
        use_staging=use_staging,
        update_horizon=1,
        observation_dtype=obs_dtype,
        renderer=renderer,
        state_dtype=state_dtype,
    )

    model_creator_template = None
    if value_function_to_share_network is not None:
        model_creator_template = value_function_to_share_network._model_creator

    # create value function
    input_shape = (None,) + obs_shape
    if value_function_name in ['vanilla', 'ensemble'] or callable(value_function_name):
        value_class = ValueVanilla if value_function_name == 'vanilla' else ValueEnsemble
        if callable(value_function_name):
            value_class = value_function_name
        value = value_class(
            sess=sess,
            replay=replay,
            obs_shape=input_shape,
            model_creator_template=model_creator_template,
            action_space_dim=env.action_space.n,
        )
    elif value_function_name == 'perfect':
        value = ValuePerfect(env=env)
    elif value_function_name == 'zero':
        value = ValueZero()
    else:
        raise RuntimeError("Unknown value function {}".format(value_function_name))

    if MPI is not None:
        value = MPIValueWrapper(value)

    if callable(agent_name):
        if use_perfect_env():
            planner = agent_name(value=value, model=model)
        else:
            planner = agent_name(value=value, model=model, true_model=true_model)
    elif agent_name == 'mcts':
        planner = MCTS(value=value, model=model)
    else:
        raise RuntimeError("Unknown step name {}".format(agent_name))

    return value, planner, env.init_kwargs
