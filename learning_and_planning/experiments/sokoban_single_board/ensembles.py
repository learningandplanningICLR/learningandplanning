"""

Vary SokobanEnvFast.seed parameter to run training on different boards.
"""


from learning_and_planning.experiments.helpers.specification_helper import \
    create_experiments_helper
from learning_and_planning.mcts.value_accumulators_ensembles import ConstKappa

base_config = {"create_agent.agent_name": "@KC_MCTS",
               "create_agent.value_function_name": "@ValueEnsemble2",
               "create_agent.replay_capacity": 100,

               "get_env_creator.env_callable_name": "@sokoban_with_finite_number_of_games",
               "SokobanEnvFast.dim_room": (10, 10),
               "SokobanEnvFast.num_boxes": 4,
               "SokobanEnvFast.penalty_for_step": 0.,
               "SokobanEnvFast.reward_box_on_target": 0.,
               "SokobanEnvFast.reward_finished": 10,
               "sokoban_with_finite_number_of_games.number_of_games": 1,

               "ValueBase.decay": 0.0,
               "ValueBase.optimizer_fn": "@tf.train.RMSPropOptimizer",
               "ValueBase.learning_rate_fn": 0.00025,
               "ValueBase.max_tf_checkpoints_to_keep": 10,
               "ValueBase.loss_fn": "@tf.losses.mean_squared_error",
               "ValueBase.loss_clip_threshold": 0.0,
               "ValueBase.activation": "identity",
               "ValueBase.model_name": "None",

               "ValueEnsemble2.num_ensemble": 20,  # These values must be equal
               "RandomGameMask.number_of_ensembles": 20,
               "BernoulliMask.number_of_ensembles": 20,
               "ConstantEnsembleMask.number_of_ensembles": 20,
               "mlp_multi_head.num_heads": 20,
               "multiple_mlps.num_heads": 20,
               "KC_MCTS.num_ensembles_per_game": 5,
               "KC_MCTS.ensemble_size": 20,
               "ValueEnsemble2.accumulator_fn": "@EnsembleValueAccumulatorMeanStdMaxUCB",
               "PoloOutOfGraphReplayBuffer.mask_game_processor_fn": "@BernoulliMask",
               "multiple_mlps.num_hidden_layers": 2,
               "mlp_multi_head.num_hidden_layers": 2,

               "ScalarValueTraits.dead_end_value": 0.,
               "MCTS.num_mcts_passes": 10,
               "MCTS.avoid_loops": True,
               "MCTS.gamma": 0.99,
               "MCTS.node_value_mode": "bootstrap",
               "MCTS.episode_max_steps": 100,
               "MCTS.avoid_history_coeff": -2,
               "Server.log_every_n": 50,
               "Server.exit_callback": "@ExitOnThreshold()",
               "ExitOnThreshold.index": 1,
               "ExitOnThreshold.threshold": 0.0001,  # just one success is enough
               "ExitOnThreshold.delay": 50,

               "Server.min_replay_history": 100,
               "PoloWrappedReplayBuffer.batch_size": 32,
               "PoloOutOfGraphReplayBuffer.solved_unsolved_ratio": 0.5,
               "training_steps": 100000,
               "train_every_num_steps": 10,
               "game_buffer_size": 25,
               "run_eval_worker": False,
               }

params_grid = {
    "SokobanEnvFast.seed": [0],  # seed determinate board used for experiment
    "EnsembleValueAccumulatorMeanStdMaxUCB.kappa_fn":
        [ConstKappa(kappa) for kappa in [3.,]],
    "BernoulliMask.p": [0.5,],
    "BernoulliMask.one_mask_per_game": [True,],
    "idx": [0,],
    "KC_MCTS.num_ensembles_per_game": [10, ],
    "ValueBase.learning_rate_fn": [0.00025,],
    "EnsembleValueAccumulatorMeanStdMaxUCB.ucb_coeff": [0.0,],
    "EnsembleValueAccumulatorMeanStdMaxUCB.exploration_target": [False, ],
    "MCTS.num_mcts_passes": [10,],
    "ValueBase.model_name": ["multiple_mlps",],
    "ValueEnsemble2.prior_scale": [None ],
}


experiments_list = create_experiments_helper(experiment_name='Sokoban single-board',
                                             python_path='.:./deps/gym-sokoban:./deps/ourlib:'
                                                         './deps/baselines:./deps/dopamine:./deps/gym-sokoban-fast:./deps/chainenv:./deps/toy-mr:',
                                             paths_to_dump='',
                                             callbacks=(),
                                             base_config=base_config, params_grid=params_grid)
