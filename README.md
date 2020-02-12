
## Uncertainty - sensitive learning and planning with ensembles

This is anonymized codebase for experiments presented in the paper "Uncertainty - sensitive learning and planning with ensembles".

## How to run it

Build singularity container

`singularity build container.simg singularity_recipe`

Run container shell

`singularity shell container.simg`

Set PYTHONPATH

`export PYTHONPATH=.:./deps/gym-sokoban:./deps/chainenv:./deps/gym-sokoban-fast:./deps/ourlib:./deps/baselines:./deps/dopamine:./deps/toy-mr`

Run experiment with

`mpirun -np 2 ./run_train_.sh learning_and_planning/experiments/chain_env/ensemble_kappa_0.py`

`-np` argument sets number of processes, in our experiments we used value `24`.

`learning_and_planning/experiments/chain_env/ensemble_kappa_0.py` is path to experiment specification file, replace it to run other experiment.

## Experiment specification files

All experiments specifications (which should be passed to `run_train_.sh`) are 
inside `learning_and_planning/experiments/` directory.

See following subdirectories for experiments presented in the paper.

* `chain_env` Deep-sea experiments.
* `toy_mr` Toy Montezuma Revenge experiments.
* `sokoban_multiboard` Sokoban experiments on multiple boards with voting.
* `sokoban_single_board` Sokoban experiments on single boards.
* `transfer` Transfer across Sokoban boards. See docstring in `transfer.py` 
for instructions.


## Train model of Sokoban environment

We provide Keras checkpoint of neural network used in our paper used to simulate Sokoban dynamics in `checkpoints/epoch.0003.hdf5`. To train MCTS-based agent using this model for planning use specification file `learning_and_planning/experiments/sokoban_multiboard/paper_exp_learned_model.py`. You can also train this model from scratch with

`python3 learning_and_planning/supervised/supervised_training.py --ex learning_and_planning/experiments/next_frame_prediction/train_sokoban_approximated_model.py`

This experiment will create new checkpoints and override existing one.

## Overview

Training is done with MPI (entry point is `learning_and_planning/mcts/mpi_train.py`). 
There is one process which performs training and logging (see `learning_and_planning/mcts/server.py`), other 
processes perform data collection (`learning_and_planning/mcts/worker.py`).


MCTS is implemented in `learning_and_planning/mcts/mcts_planner.py` and 
`learning_and_planning/mcts/mcts_planner_with_voting.py`, see experiment 
specification files to determine which class is used for given experiment (parameter 
`create_agent.agent_name`).

Criterions for action choice methods based on ensembles can be found in `value_accumulators_ensembles.py`.

For Neural Network architectures see `learning_and_planning/mcts/mcts_planner.py`.

Training masks are implemented in `learning_and_planning/mcts/mask_game_processors.py`. 

