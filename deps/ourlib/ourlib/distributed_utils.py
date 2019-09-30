from collections import namedtuple

import numpy as np

TAGS = namedtuple('TAGS', ['GAME', 'EXIT', 'PARAMETERS', 'GIN', 'NEPTUNE_EXP_METADATA'])(0, 1, 2, 3, 4)


def get_mpi_rank_or_0():
    try:
        from mpi4py import MPI
        if MPI is not None:
            return MPI.COMM_WORLD.Get_rank()
        else:
            return 0
    except ImportError:
        return 0


def get_mpi_comm_world():
    from mpi4py import MPI
    return MPI.COMM_WORLD


def get_mpi_comm_self():
    from mpi4py import MPI
    return MPI.COMM_SELF


def is_mpi_enabled():
    try:
        from mpi4py import MPI
        if MPI is not None:
            return True
        else:
            return False
    except ImportError:
        return False


def mpi_Send_string(s, dest, tag):
    comm = get_mpi_comm_world()
    encoded_s = np.array([ord(c) for c in s], dtype=np.int)
    size = encoded_s.size
    comm.Send(np.array([size], dtype=np.int), dest=dest, tag=tag)
    comm.Send(encoded_s, dest=dest, tag=tag)


def mpi_Recv_string(source, tag):
    comm = get_mpi_comm_world()
    size = np.array([-1], dtype=np.int)
    comm.Recv(size, source=source, tag=tag)
    message_array = np.zeros(size, dtype=np.int)
    comm.Recv(message_array, source=source, tag=tag)
    s = "".join([chr(c) for c in message_array])
    return s
