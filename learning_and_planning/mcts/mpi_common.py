# Changing this is risky! This is convention used in RL MPI training code.
# Somewhere in the code raw numbers might be used instead of these aliases, so
# be prepared for problems when changing values below.
SERVER_RANK = 0
EVALUATOR_RANK = 1  # optional, this rank might be also used for normal Worker.
