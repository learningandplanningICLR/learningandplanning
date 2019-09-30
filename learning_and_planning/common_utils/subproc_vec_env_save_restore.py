import traceback
from multiprocessing import Process, Pipe

import numpy as np

from baselines.common.vec_env import VecEnv, CloudpickleWrapper

from learning_and_planning.mcts.value import ValueDictionary


def worker(remote, parent_remote, env_fn_wrapper):
    try:
        parent_remote.close()
        env = env_fn_wrapper.x()
        try:
            while True:
                cmd, data = remote.recv()
                if cmd == 'step':
                    ob, reward, done, info = env.step(data)
                    # if done:
                    #  ob = env.reset()
                    remote.send((ob, reward, done, info))
                elif cmd == 'reset':
                    ob = env.reset()
                    remote.send(ob)
                elif cmd == 'render':
                    remote.send(env.render(**data))
                elif cmd == 'close':
                    remote.close()
                    break
                elif cmd == 'clone':
                    remote.send(env.clone_full_state())
                elif cmd == 'restore':
                    env.restore_full_state(data)
                elif cmd == 'recover_state':
                    remote.send(env.recover_state(data))
                elif cmd == 'get_spaces':
                    remote.send((env.observation_space, env.action_space))
                elif cmd == 'dim_room':
                    remote.send(env.dim_room)
                elif cmd == 'current_seed':
                    remote.send(env.current_seed)
                elif cmd == 'init_kwargs':
                    remote.send(env.init_kwargs)
                elif cmd == 'setattr':
                    field, value = data
                    print("Setattr:{}".format(data))
                    setattr(env, field, value)
                    remote.send(0)
                else:
                    raise NotImplementedError
        except KeyboardInterrupt:
            print('SubprocVecEnv worker: got KeyboardInterrupt')
        finally:
            env.close()
    except Exception as e:
        print('Worker Died!\n')
        print(traceback.format_exc())


class SubprocVecEnvCloneRestore(VecEnv):
    """
  VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
  Recommended to use when num_envs > 1 and step() can be a bottleneck.
  """

    def __init__(self, env_fns, data=None, deadlock_reward=-10):
        """
    Arguments:

    env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
    """
        self.waiting = False
        self.closed = False
        self.nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])

        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(('dim_room', None))
        self.dim_room = self.remotes[0].recv()
        self.viewer = None

        self.read_from_data = data is not None
        if self.read_from_data:
            self.values = []
            self.roots = []
            for d in data:
                vf = ValueDictionary()
                vf.load_vf_for_root(d, compressed=True)
                self.values.append(vf)
                self.roots.append(vf.root)
            self.num_levels = len(data)
            self.current_room_id = -1
            self.deadlock_reward = deadlock_reward

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        for info in infos:
            info['deadlock'] = False
        if self.read_from_data:
            states = self.clone_full_state()
            for idx, state in enumerate(states):
                if self.values[self.current_room_id](states=state) == -np.inf:
                    rews[idx] += self.deadlock_reward
                    infos['deadlock'][idx] = True
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, how_many=None):
        self._assert_not_closed()
        last = self.nenvs if not how_many else min(self.nenvs, how_many)
        if self.read_from_data:
            self.current_room_id = (self.current_room_id + 1) % self.num_levels
            root = self.roots[self.current_room_id]
            self.restore_full_state([root]*self.num_envs, how_many)
            obs = self.render(how_many)
        else:
            for remote in self.remotes[:last]:
                remote.send(('reset', None))
            obs = np.stack([remote.recv() for remote in self.remotes[:last]])
        return obs

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def render(self, how_many=None, **kwargs):
        self._assert_not_closed()
        last = self.nenvs if not how_many else min(self.nenvs, how_many)
        for pipe in self.remotes[:last]:
            pipe.send(('render', kwargs))
        imgs = [pipe.recv() for pipe in self.remotes[:last]]
        return np.array(imgs)

    def restore_full_state(self, states, how_many=None):
        self._assert_not_closed()
        last = self.nenvs if not how_many else min(self.nenvs, how_many)
        for remote, state in zip(self.remotes[:last], states[:last]):
            remote.send(('restore', state))
        self.waiting = True

    def clone_full_state(self, how_many=None):
        self._assert_not_closed()
        last = self.nenvs if not how_many else min(self.nenvs, how_many)
        for remote in self.remotes[:last]:
            remote.send(('clone', None))
        states = [remote.recv() for remote in self.remotes[:last]]
        return states

    def recover_state(self, observations):
        self._assert_not_closed()
        last = len(observations)
        for remote, obs in zip(self.remotes[:last], observations[:last]):
            remote.send(('recover_state', obs))
        states = [remote.recv() for remote in self.remotes[:last]]
        return states

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def pm_setattr(self, field_name, value):
        print("pm_setattr")
        for remote in self.remotes:
            remote.send(('setattr', [field_name, value]))
        [remote.recv() for remote in self.remotes]

    @property
    def current_seed(self):
        self.remotes[0].send(('current_seed', None))
        return self.remotes[0].recv()

    @property
    def inner_env_init_kwargs(self):
        self.remotes[0].send(('init_kwargs', None))
        return self.remotes[0].recv()
