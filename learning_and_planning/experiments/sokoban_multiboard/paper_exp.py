from ourlib.rl_misc import LinearlyDecayingEpsilon
from learning_and_planning.experiments.helpers.specification_helper import \
    create_experiments_helper

base_config = {"create_agent.agent_name": "@MCTSWithVoting",
               "create_agent.value_function_name": "ensemble",
               "create_agent.replay_capacity": 10000,

               "get_env_creator.env_callable_name": "gym_sokoban_fast:SokobanEnvFast",
               "get_env_creator.dim_room": (10, 10),
               "get_env_creator.num_boxes": 4,
               "get_env_creator.num_gen_steps": None,
               "get_env_creator.mode": "one_hot",
               # SPARSE
               "get_env_creator.penalty_for_step": 0.,
               "get_env_creator.reward_box_on_target": 0.,
               "get_env_creator.reward_finished": 10.,

               "ValueBase.decay": 0.1,
               "ValueBase.optimizer_fn": "@tf.train.RMSPropOptimizer",
               "ValueBase.learning_rate_fn": None,
               "ValueBase.max_tf_checkpoints_to_keep": 10,
               "ValueBase.loss_fn": "@tf.losses.mean_squared_error",
               "ValueBase.loss_clip_threshold": 0.0,
               "ValueBase.activation": "identity",

               "ValueEnsemble.include_prior": True,  # info: similar to no-ensemble case
               "ValueEnsemble.accumulator_fn": "@EnsembleValueAccumulatorVoting",
               "EnsembleValueAccumulatorVoting.bonus_fn": "@ucb1",  # ucb1

               "MCTSWithVoting.num_mcts_passes": 10,
               "MCTSWithVoting.num_sampling_moves": 0,
               "MCTSWithVoting.value_annealing": 1.0,
               "MCTSWithVoting.avoid_loops": True,
               "MCTSWithVoting.node_value_mode": "factual_rewards",
               "MCTSWithVoting.differ_final_rating": False,

               "Server.min_replay_history": 1000,
               "PoloOutOfGraphReplayBuffer.solved_unsolved_ratio": 0.5,
               "curriculum": False,
               "training_steps": 500000,
               "train_every_num_steps": 100,
               "game_buffer_size": 25,
               "Server.log_every_n": 50,
               "run_eval_worker": False,
               #"EvaluatorWorker.eval_planners_params": {'eval_10_mcts': {"MCTSWithVoting.num_mcts_passes": 10}},
               }


params_grid = {"get_env_creator.seed": [None],  # 1230
               "ValueBase.model_name": ["convnet_mnist"],  #, "convnet_mnist_bottleneck"],
               "ValueBase.decay": [0.1],
               "ValueEnsemble.num_ensemble___": [3],  
               "PoloWrappedReplayBuffer.batch_size___": [96],  # 32 * 4
               "ValueEnsemble.prior_scale": [0.1],
               "EnsembleValueAccumulatorVoting.bonus_loading": [0.],  # info: similar to no-ensemble case
               "MCTSWithVoting.gamma": [0.99],
               "MCTSWithVoting.avoid_history_coeff": [-0.5],
               "EnsembleValueTraits.dead_end_value": [-0.1],
               "MCTSWithVoting.episode_max_steps": [200],
               "ValueBase.learning_rate_fn": [LinearlyDecayingEpsilon(lr=0.00025,
                                                                      decay_period=2000.0 * 100,
                                                                      warmup_steps=100 * 100,
                                                                      epsilon=0.01)],
               }

experiments_list = create_experiments_helper(experiment_name='sokoban_ensembles',
                                             python_path='.:./deps/gym-sokoban:./deps/ourlib:'
                                                         './deps/baselines:./deps/dopamine:./deps/gym-sokoban-fast',
                                             paths_to_dump='',
                                             base_config=base_config, params_grid=params_grid)
