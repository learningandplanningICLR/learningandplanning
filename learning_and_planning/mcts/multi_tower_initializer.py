import tensorflow as tf
from tensorflow.python.pywrap_tensorflow_internal import NewCheckpointReader

from ourlib.distributed_utils import get_mpi_comm_world
from learning_and_planning.mcts.mpi_common import SERVER_RANK


def create_multihead_initializers(checkpoint_paths):
    comm = get_mpi_comm_world()
    rank = comm.Get_rank()
    if rank != SERVER_RANK:
        return []

    readers = [NewCheckpointReader(checkpoint_path) for checkpoint_path in checkpoint_paths]
    variables_name = [var_name for var_name in list(readers[0].get_variable_to_shape_map().keys()) if "RMSProp" not in var_name]
    truncated_variables_name = [var_name[14:] for var_name in variables_name]

    initializers_ = []
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        for idx, reader in enumerate(readers):
            for var_name in truncated_variables_name:
                check_point_var_name = "model/tower_0/" + var_name
                learnable_ensemble_var_name = f"model/tower_{idx}/" + var_name
                values_np = reader.get_tensor(check_point_var_name)
                # print("var_name", var_name)
                # print("values_np", values_np)
                initializers_.append(tf.get_variable(learnable_ensemble_var_name).assign(values_np))

    return initializers_
