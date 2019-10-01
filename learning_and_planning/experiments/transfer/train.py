from learning_and_planning.experiments.helpers.specification_helper import \
    create_experiments_helper

base_config = {"create_agent.agent_name": "mcts",
               "create_agent.value_function_name": "@ValueEnsemble2",
               "create_agent.replay_capacity": 100000,

               "get_env_creator.env_callable_name": "@sokoban_with_finite_number_of_games",
               "SokobanEnvFast.dim_room": (10, 10),
               "SokobanEnvFast.num_boxes": 4,
               "SokobanEnvFast.penalty_for_step": 0.,
               "SokobanEnvFast.reward_box_on_target": 0.,
               "SokobanEnvFast.reward_finished": 10,

               "sokoban_with_finite_number_of_games.number_of_games": 1,

               "ValueBase.decay": 0.01,
               "ValueBase.optimizer_fn": "@tf.train.RMSPropOptimizer",
               "ValueBase.learning_rate_fn": 0.00025,
               "ValueBase.max_tf_checkpoints_to_keep": 10,
               "ValueBase.loss_fn": "@tf.losses.mean_squared_error",
               "ValueBase.accumulator_fn": "@EnsembleValueAccumulatorMeanVarMaxUCB",
               "ValueBase.loss_clip_threshold": 0.0,
               "ValueBase.activation": "identity",
               "ValueBase.model_name": "convnet_mnist_multi_towers",
               "convnet_mnist_multi_towers.tower_depth": 5,

               "EnsembleConfigurator.num_ensembles": 1,

               "ValueEnsemble2.accumulator_fn": "@EnsembleValueAccumulatorMeanVarMaxUCB",
               "ValueEnsemble2.prior_scale": None,
               "PoloOutOfGraphReplayBuffer.mask_game_processor_fn": "@RandomEnsembleTupleMaskProcessor",

               "ScalarValueTraits.dead_end_value": -2.0,
               "MCTSValue.num_mcts_passes": 10,
               "MCTSValue.num_sampling_moves": 0,
               "MCTSValue.avoid_loops": True,
               "MCTSValue.gamma": 0.99,
               "MCTSValue.node_value_mode": "factual_rewards",
               "MCTSValue.episode_max_steps": 100,
               "MCTSValue.history_process_fn": "@sokoban_hindsight",
               "sokoban_hindsight.intensity": 0.2,
               "MCTSValue.avoid_history_coeff": -2,
               "Server.log_every_n": 50,
               "Server.save_checkpoint_every_train_steps": 1,
               "Server.exit_callback": "@exit_on_threshold",
               "exit_on_threshold.index": 1,
               "exit_on_threshold.threshold": 0.9,

               "Server.min_replay_history": 100,
               "PoloWrappedReplayBuffer.batch_size": 32,
               "PoloOutOfGraphReplayBuffer.solved_unsolved_ratio": 0.5,
               "training_steps": 100000,
               "train_every_num_steps": 100,
               "game_buffer_size": 25,
               "run_eval_worker": False,
               }


params_grid = {"SokobanEnvFast.seed": [0,]}

experiments_list = create_experiments_helper(experiment_name='Transfer, train',
                                             python_path='.:./deps/gym-sokoban:./deps/ourlib:'
                                                         './deps/baselines:./deps/dopamine:./deps/gym-sokoban-fast',
                                             paths_to_dump='',
                                             base_config=base_config, params_grid=params_grid)
