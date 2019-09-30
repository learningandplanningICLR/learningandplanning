import copy
import time
from multiprocessing import Process, Pipe
from typing import NamedTuple, Deque, List, Optional

import numpy as np

from baselines.common.vec_env import VecEnv, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'call_method':
            (method_name, kwargs) = data
            getattr(env, method_name)(**kwargs)
        else:
            raise NotImplementedError


RemoteConnection = object
Observation = object


class ReadyEnv(NamedTuple):
    remote: RemoteConnection
    initial_obs: object


class RoundRobinSubprocVecEnv(VecEnv):
    ready_waiting_envs: Deque[ReadyEnv] = Deque[ReadyEnv]()
    resetting_envs: List[RemoteConnection] = []
    active_envs: List[Optional[RemoteConnection]]
    initial_obs: List[Optional[Observation]]

    def __init__(self, env_fn, observation_space, action_space, num_envs: int, extra_envs: int = 1):
        """
        envs: list of gym environments to run in subprocesses
        """
        VecEnv.__init__(self, num_envs, observation_space, action_space)

        self.waiting = False
        self.closed = False
        remotes, work_remotes = zip(*[Pipe() for _ in range(num_envs + extra_envs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote) in zip(work_remotes, remotes)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in work_remotes:
            remote.close()

        self.active_envs = [None for _ in range(num_envs)]
        self.initial_obs = [None for _ in range(num_envs)]

        for remote in remotes:
            remote.send(('reset', None))
            self.resetting_envs.append(remote)

        self._wait_for_active_envs()

    def step(self, actions):
        for remote, action in zip(self.active_envs, actions):
            remote.send(('step', action))

        results = []

        for env_ix in range(self.num_envs):
            remote = self.active_envs[env_ix]
            result = remote.recv()  # sometimes it receives reset frame!
            results.append(result)
            _, _, done, _ = result

            if done:
                remote.send(('reset', None))
                self.active_envs[env_ix] = None
                self.resetting_envs.append(remote)

        obs, rews, dones, infos = zip(*results)
        obs = np.stack(obs)
        rews = np.stack(rews)
        dones = np.stack(dones)

        if np.any(dones):
            self._wait_for_active_envs()
            initial_obs = np.stack(self.initial_obs)
            obs[dones] = initial_obs[dones]

        return obs, rews, dones, infos

    def _wait_for_active_envs(self):
        while None in self.active_envs:
            self._check_env_statuses_and_update()
            time.sleep(0.05)

    def _check_env_statuses_and_update(self):
        resetting_envs = copy.copy(self.resetting_envs)
        for resetting_env in resetting_envs:
            if resetting_env.poll():
                data = resetting_env.recv()
                self.ready_waiting_envs.append(ReadyEnv(resetting_env, data))
                self.resetting_envs.remove(resetting_env)

        for env_ix in range(self.num_envs):
            if self.active_envs[env_ix] is None and len(self.ready_waiting_envs) > 0:
                ready_env = self.ready_waiting_envs.pop()
                self.active_envs[env_ix] = ready_env.remote
                self.initial_obs[env_ix] = ready_env.initial_obs

    def _wait_until_all_resetting_envs_are_ready(self):
        while len(self.resetting_envs) > 0:
            self._check_env_statuses_and_update()
            time.sleep(0.05)

    def step_async(self, actions):
        "Dummy method - all logic is implemented in step()"
        pass

    def step_wait(self):
        "Dummy method - all logic is implemented in step()"
        pass

    def reset(self, **kwargs):
        self._wait_until_all_resetting_envs_are_ready()  # so we don't send new reset while waiting for old one
        all = self._get_all_envs()
        for remote in all:
            remote.send(('reset', kwargs))
        return np.stack([remote.recv() for remote in all])[:self.num_envs]

    def call_method(self, method_name, **kwargs):
        for remote in self._get_all_envs():
            remote.send(('call_method', (method_name, kwargs)))

    def _get_all_envs(self):
        active_envs = list(filter(lambda env: env is not None, self.active_envs))
        return [*active_envs, *map(lambda e: e.remote, self.ready_waiting_envs), *self.resetting_envs]

    def close(self):
        raise Exception("TODO")
