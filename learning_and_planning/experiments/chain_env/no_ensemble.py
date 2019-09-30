from learning_and_planning.experiments.helpers.specification_helper import \
    create_experiments_helper
from learning_and_planning.mcts.value_accumulators_ensembles import ConstKappa


base_config = {"create_agent.agent_name": "@KC_MCTSValue",
               "create_agent.value_function_name": "@ValueEnsemble2",
               "create_agent.replay_capacity": 100,

               "get_env_creator.env_callable_name": "chain_env:ChainEnvironment", # "gym_sokoban_fast:SokobanEnvFast",

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
               "linear_multi_head.num_heads": 1,
               "KC_MCTSValue.num_ensembles_per_game": 1,
               "KC_MCTSValue.ensemble_size": 1,
               "ValueEnsemble2.accumulator_fn": "@EnsembleValueAccumulatorMeanStdMaxUCB",
               "PoloOutOfGraphReplayBuffer.mask_game_processor_fn": "@RandomGameMask",

               "ScalarValueTraits.dead_end_value": 0.,
               "MCTSValue.num_mcts_passes": 10,
               "MCTSValue.avoid_loops": False,
               "MCTSValue.gamma": 0.99,
               "MCTSValue.node_value_mode": "bootstrap",
               "MCTSValue.episode_max_steps": 100,
               "MCTSValue.avoid_history_coeff": 0,
               "MCTSValue.num_sampling_moves": 0,

               "Server.min_replay_history": 100,
               "Server.log_every_n": 10,
               "PoloWrappedReplayBuffer.batch_size": 32,
               "PoloOutOfGraphReplayBuffer.solved_unsolved_ratio": 0.5,
               "training_steps": 200000,
               "train_every_num_steps": 10,
               "game_buffer_size": 10,
               "run_eval_worker": False,
               }


params_grid = {
    "get_env_creator.N": [40],  # Here set problem size
    "MCTSValue.episode_max_steps": [51, ],  # should be no less than N for ChainEnvironment
    "EnsembleValueAccumulatorMeanStdMaxUCB.kappa_fn": [ConstKappa(0),],
    "ValueBase.model_name": ["linear_multi_head",],
}

experiments_list = create_experiments_helper(experiment_name='Ensemble ChainEnv',
                                             python_path='.:./deps/gym-sokoban:./deps/ourlib:'
                                                         './deps/baselines:./deps/dopamine:./deps/gym-sokoban-fast:./deps/chainenv:',
                                             paths_to_dump='',
                                             exclude=[],
                                             base_config=base_config, params_grid=params_grid)
