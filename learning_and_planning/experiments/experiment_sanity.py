from learning_and_planning.experiments.helpers.specification_helper import \
    create_experiments_helper

base_config = {"create_agent.agent_name": "mcts",
               "create_agent.value_function_name": "vanilla",
               "create_agent.replay_capacity": 10000,

               "get_env_creator.env_callable_name": "gym_sokoban_fast:SokobanEnvFast",
               "SokobanEnvFast.dim_room": (8, 8),
               "SokobanEnvFast.num_boxes": 2,
               "SokobanEnvFast.mode": "one_hot",

               "ValueBase.decay": 0.1,
               "ValueBase.optimizer_fn": "@tf.train.RMSPropOptimizer",
               "ValueBase.learning_rate_fn": 0.00025,
               "ValueBase.max_tf_checkpoints_to_keep": 10,
               "ValueBase.loss_fn": "@tf.losses.mean_squared_error",
               "ValueBase.accumulator_fn": "@ScalarValueAccumulator",
               "ValueBase.loss_clip_threshold": 0.0,
               "ValueBase.activation": "identity",
               "ValueBase.model_name": "convnet_mnist",
               "ScalarValueTraits.dead_end_value": -2.0,
               "MCTS.num_mcts_passes": 10,
               "MCTS.num_sampling_moves": 0,
               "MCTS.value_annealing": 1.0,
               "MCTS.avoid_loops": True,
               "MCTS.gamma": 0.99,
               "MCTS.node_value_mode": "bootstrap",
               "MCTS.episode_max_steps": 50,

               "Server.min_replay_history": 1000,
               "PoloWrappedReplayBuffer.batch_size": 32,
               "PoloOutOfGraphReplayBuffer.solved_unsolved_ratio": 0.5,
               "curriculum": False,
               "training_steps": 500000,
               "train_every_num_steps": 100,
               "game_buffer_size": 25,
               "log_every_n": 50,
               "run_eval_worker": False,
               "Server.save_checkpoint_every_train_steps": 500,
               }


params_grid = {"MCTS.avoid_loops": [True, False], "ValueBase.model_name": ["convnet_mnist",
                                                                                "kc_parametrized_cnn_v0_2"]}

experiments_list = create_experiments_helper(experiment_name='Mcts sanity experiment',
                                             python_path='.:./deps/gym-sokoban:./deps/ourlib:'
                                                         './deps/baselines:./deps/dopamine:./deps/gym-sokoban-fast',
                                             paths_to_dump='',
                                             base_config=base_config, params_grid=params_grid)
