from learning_and_planning.experiments.helpers.specification_helper import \
    create_experiments_helper

base_config = {"create_agent.agent_name": "mcts",
               "create_agent.value_function_name": "vanilla",
               "create_agent.replay_capacity": 100000,

               "get_env_creator.env_callable_name": "gym_sokoban_fast:SokobanEnvFast",
               "SokobanEnvFast.dim_room": (10, 10),
               "SokobanEnvFast.num_boxes": 4,
               "SokobanEnvFast.mode": "one_hot",
               "SokobanEnvFast.penalty_for_step": 0.,
               "SokobanEnvFast.reward_box_on_target": 0.,
               "SokobanEnvFast.reward_finished": 10.,

               "ValueBase.decay": 0.05,
               "ValueBase.optimizer_fn": "@tf.train.RMSPropOptimizer",
               "ValueBase.learning_rate_fn": 0.00025,
               "ValueBase.max_tf_checkpoints_to_keep": 10,
               "ValueBase.loss_fn": "@tf.losses.mean_squared_error",
               "ValueBase.accumulator_fn": "@ScalarValueAccumulator",
               "ValueBase.loss_clip_threshold": 0.0,
               "ValueBase.activation": "identity",
               "ValueBase.model_name": "convnet_mnist",
               "ScalarValueTraits.dead_end_value": -2.0,
               "MCTSValue.num_mcts_passes": 10,
               "MCTSValue.num_sampling_moves": 0,
               "MCTSValue.avoid_loops": True,
               "MCTSValue.gamma": 0.99,
               "MCTSValue.episode_max_steps": 200,
               "MCTSValue.avoid_history_coeff": -2.,

               "Server.min_replay_history": 1000,
               "PoloOutOfGraphReplayBuffer.solved_unsolved_ratio": 0.5,
               "training_steps": 5000000,
               "train_every_num_steps": 100,
               "game_buffer_size": 25,
               "run_eval_worker": False,
               "Server.log_every_n": 50,
               }


params_grid = {"MCTSValue.node_value_mode": ["factual_rewards"],
               "PoloWrappedReplayBuffer.batch_size": [32],
               "SokobanEnvFast.penalty_for_step___": [0.],
               "SokobanEnvFast.reward_box_on_target___": [0.],
               }

experiments_list = create_experiments_helper(experiment_name='sokoban_no_ensembles',
                                             python_path='.:./deps/gym-sokoban:./deps/ourlib:'
                                                         './deps/baselines:./deps/dopamine:./deps/gym-sokoban-fast',
                                             paths_to_dump='',
                                             base_config=base_config, params_grid=params_grid)
