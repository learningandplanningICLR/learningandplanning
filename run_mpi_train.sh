#!/usr/bin/env bash

if test "$#" -eq 1; then
    EXPERIMENT=$1
else
    echo "Pass one experiment"
    exit 1
fi

echo "========================================="
printf "\033[0;31mWill run $EXPERIMENT\033[0m\n"
echo "========================================="

export LOCAL_EXP=True
mpirun -np 2 ./run_train_.sh $EXPERIMENT
