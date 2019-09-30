
## How to run it
Install openmpi >= 3.0.0, create virtualenv with python 3.6.

Install modules  from requirements_cpu.txt.

`pip install -r requirements_cpu.txt`


Set PYTHONPATH

`export PYTHONPATH=.:./deps/gym-sokoban:./deps/chainenv:./deps/gym-sokoban-fast:./deps/ourlib:./deps/baselines:./deps/dopamine:./deps/toy-mr`

Run experiment with

`mpirun -np 2 ./run_train_.sh learning_and_planning/experiments/chain_env/ensemble_kappa_0.py`

`-np` argument sets number of processes, in our experiments we used value `24`, 
but this requires appropriate number of cores available.

## Experiment files

All experiments specifications (which should be passed to `run_train_.sh`) are 
inside learning_and_planning/experiments/ directory.

Deep-sea experiments can be run with specifications from `chain_env` subdirectory.

For multiboard-sokoban experiments see subdirectory `multiboard-sokoban`.


