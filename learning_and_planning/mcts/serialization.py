import numpy as np
import collections

# INFO: similar to dopamine
SerializedVariable = (
    collections.namedtuple('shape_type', ['name', 'shape', 'dtype'])
)
BatchedVariable = (
    collections.namedtuple('shape_type', ['name', 'shape', 'byte_size', 'dtype'])
)

class Serializer(object):
    def __init__(self):
        self.variables = []
        self.batched_variables = []
        self.buffer_size = 0
        self.vsizes = dict()  # variable -> byte size

    def add_variable(self, name, shape, dtype):
        self.variables.append(
            SerializedVariable(name, shape, dtype)
        )
        self.vsizes[name] = 0
        for sh, dt in zip(shape[1:], dtype):
            byte_size = shape[0] * np.prod(sh) * dt().itemsize
            self.buffer_size += byte_size
            self.vsizes[name] += byte_size
            if not isinstance(sh, tuple):
                sh = (sh, )
            self.batched_variables.append(
                BatchedVariable(name,
                                (shape[0],) + sh,
                                byte_size,
                                dt,
                )
            )

    def serialize(self, **data_kwargs):
        buffer = []
        for variable in self.variables:
            var_size = 0
            name = variable.name
            dtype = variable.dtype
            data = data_kwargs[name]
            if not isinstance(data, list):
                data = [(data, )]

            padding = variable.shape[0] - len(data)
            if padding > 0:
                data += [data[-1]] * padding

            for chunk, dt in zip(list(zip(*data)), dtype):  # chunk could be all states, all actions, etc.
                chunk_bytes = np.stack(chunk).astype(dt).tobytes()
                var_size += len(chunk_bytes)
                buffer.append(chunk_bytes)
            if var_size != self.vsizes[name]:
                raise ValueError(
                    f"Expected {self.vsizes[name]} bytes for serialized "
                    f"variable '{name}', got {var_size} instead."
                )

        result = b''.join(buffer)

        return np.frombuffer(result, dtype=np.uint8)

    def deserialize(self, data):
        buffer = data.tobytes()
        tmp = {}
        pos = 0
        for variable in self.batched_variables:
            name = variable.name
            dtype = variable.dtype
            shape = variable.shape
            byte_size = variable.byte_size

            chunk = buffer[pos:pos+byte_size]
            chunk = np.frombuffer(chunk, dtype=dtype).reshape(shape)

            if name not in tmp:
                tmp[name] = []
            tmp[name].append(chunk)

            pos += byte_size

        data = {}
        for variable in self.variables:
            name = variable.name
            chunk = [tuple([l if len(l) > 1 else l.item() for l in lst]) for lst in zip(*tmp[name])]
            data[name] = chunk if len(chunk) > 1 else chunk[0][0]

        return data



if __name__ == "__main__":
    from learning_and_planning.envs.sokoban_env_creator import get_env_creator

    env = get_env_creator("gym_sokoban.envs:SokobanEnv",
                          num_envs=1,
                          dim_room=(6,6),
                          num_boxes=1,
                          max_steps=50,
                          mode='one_hot',
                          reward_shaping="dense",
                          num_gen_steps=27,
                          seed=123,
                          verbose=False,
                          )()

    env.reset()

    o = None
    r = 0.
    d = False
    data = []
    while not d:
        a = np.random.randint(4)
        o, r, d, _ = env.step(a)
        s = env.clone_full_state()
        data.append((s,a))

    ser = Serializer()
    ser.add_variable("game", shape=(50, len(data[0][0]), 1), dtype=[np.uint64, np.int32])

    data_kwargs = {"game": data}

    s = ser.serialize(**data_kwargs)
    d = ser.deserialize(s)['game']

    print([a for _, a in data])
    print([a for _, a in d])

    print(data[0][0])
    print(d[0][0])
