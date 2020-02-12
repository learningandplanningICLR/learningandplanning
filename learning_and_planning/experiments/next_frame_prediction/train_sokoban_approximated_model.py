from learning_and_planning.experiments.helpers.specification_helper import \
    create_experiments_helper

base_config = dict(
    data_files_prefix= "NOT_USED",
    env_dim_room=(8, 8), env_num_boxes=4, env_mode="one_hot",
    n_cores=24,
    target="vf_positive_only",
    # Network parameters
    net_name="next_frame_and_done_cnn",
    net_cnn_l2=0., net_cnn_channels=128, net_cnn_n_layers=2,
    net_cnn_final_pool_size=(1, 1),
    net_cnn_batch_norm=False,
    net_global_average_pooling=True,  # done after conv, before fully connected
    net_fc_n_hidden=128, net_fc_n_layers=0,
    net_fc_dropout=0, net_fc_l2=0.,
    net_fc_dropout_input=False,
    net_image_output=False,
    lr=0.0002, lr_decay=0.1,
    loss=None,
    # Validation
    histogram_freq=None, epochs=100, batch_size=50,
    eval_games_to_play=0,
    shards_to_use=4000,
    validation_shards=100,
    # shards_to_use=2,  # debug settings
    # validation_shards=1,  # debug settings
    save_every=1,
    sample_data=True, max_samples_per_board=100,
    validate_every_batch=50000,
    neptune_first_batch=5000
)


params_grid = {
    "target": ["next_frame_and_done",],  # ["vf_and_type", "vf", "best_action"],
    "lr": [0.00005,],
    "net_cnn_n_layers": [2],
    "net_cnn_channels": [64],
    "net_image_output": [True,],
    "env_dim_room": [(10, 10),],
    "env_mode": ["one_hot",],
    "net_cnn_kernel_size": [(5, 5),],
}


experiments_list = create_experiments_helper(experiment_name='Next frame sokoban',
                                             python_path='.:./deps/gym-sokoban:./deps/ourlib:'
                                                         './deps/baselines:./deps/dopamine:./deps/gym-sokoban-fast:./deps/chainenv:./polo_plus/kc:./deps/toy-mr:',
                                             paths_to_dump='',
                                             exclude=[],
                                             base_config=base_config, params_grid=params_grid)
