#!/usr/bin/env bash

echo $OMPI_COMM_WORLD_RANK
python3 learning_and_planning/mcts/mpi_train.py --ex $1
