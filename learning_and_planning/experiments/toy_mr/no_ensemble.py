from learning_and_planning.experiments.helpers.specification_helper import \
    create_experiments_helper

from learning_and_planning.mcts.value_accumulators_ensembles import ConstKappa


base_config = {"create_agent.agent_name": "@KC_MCTS",
               "create_agent.value_function_name": "@ValueEnsemble2",
               "create_agent.replay_capacity": 100,

               "get_env_creator.env_callable_name": "toy_mr:ToyMR", # "gym_sokoban_fast:SokobanEnvFast",

               "ValueBase.decay": 0.0,
               "ValueBase.optimizer_fn": "@tf.train.RMSPropOptimizer",
               "ValueBase.learning_rate_fn": 0.00025,
               "ValueBase.max_tf_checkpoints_to_keep": 10,
               "ValueBase.loss_fn": "@tf.losses.mean_squared_error",
               "ValueBase.accumulator_fn": "@ScalarValueAccumulator",
               "ValueBase.loss_clip_threshold": 0.0,
               "ValueBase.activation": "identity",
               "ValueBase.model_name": "convnet_mnist",

               "ValueEnsemble2.num_ensemble": 1,  # These values must be equal
               "RandomGameMask.number_of_ensembles": 1,
               "ConstantEnsembleMask.number_of_ensembles": 1,
               "mlp_multi_head.num_heads": 1,
               "multiple_mlps.num_heads": 1,
               "KC_MCTS.num_ensembles_per_game": 1,
               "KC_MCTS.ensemble_size": 1,
               "ValueEnsemble2.accumulator_fn": "@EnsembleValueAccumulatorMeanStdMaxUCB",
               "PoloOutOfGraphReplayBuffer.mask_game_processor_fn": "@RandomGameMask",

               "ScalarValueTraits.dead_end_value": 0.,
               "MCTS.num_mcts_passes": 10,
               "MCTS.avoid_loops": False,
               "MCTS.gamma": 0.99,
               "MCTS.node_value_mode": "bootstrap",
               "MCTS.episode_max_steps": 100,
               "MCTS.avoid_history_coeff": 0,
               "MCTS.num_sampling_moves": 0,

               "Server.min_replay_history": 100,
               "PoloWrappedReplayBuffer.batch_size": 32,
               "PoloOutOfGraphReplayBuffer.solved_unsolved_ratio": 0.5,
               "training_steps": 200000,
               "train_every_num_steps": 10,
               "game_buffer_size": 25,
               "run_eval_worker": False,
               }


params_grid = {
    "idx": [0,],
    "ValueBase.learning_rate_fn": [0.00025,],
    "MCTS.episode_max_steps": [300,],
    "MCTS.avoid_history_coeff": [-0.2,],
    "MCTS.avoid_loops": [True,],
    "get_env_creator.map_file": ["deps/toy-mr/toy_mr/mr_maps/full_mr_map.txt"],
    "get_env_creator.absolute_coordinates": [False,],
    "get_env_creator.trap_reward": [0,],
    "get_env_creator.doors_keys_scale": [1,],
    "get_env_creator.save_enter_cell": [False,],
    "MCTS.num_mcts_passes": [10,],
    "ValueBase.model_name": ["multiple_mlps",],
    "ValueEnsemble2.prior_scale": [0.,],
}

experiments_list = create_experiments_helper(experiment_name='Toy MR',
                                             python_path='.:./deps/gym-sokoban:./deps/ourlib:'
                                                         './deps/baselines:./deps/dopamine:./deps/gym-sokoban-fast:./deps/chainenv:./deps/toy-mr:',
                                             paths_to_dump='',
                                             base_config=base_config,
                                             params_grid=params_grid)
