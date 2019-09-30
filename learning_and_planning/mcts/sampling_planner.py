import gin
import numpy as np

from learning_and_planning.mcts.levin_planner import softmax


@gin.configurable
class SamplingPlanner:
    def __init__(self,
                 model,
                 value,
                 episode_max_steps=50,
                 temperature=1.,
                 **kwargs):
        super().__init__()
        self._value = value  # generalized value, e.g. could be ensemble
        self._model = model
        self.episode_max_steps = episode_max_steps
        self.temperature = temperature

    def init(self):
        obs = self._model.reset()
        state = self._model.state()
        return obs, state

    def run_one_episode(self):
        obs, state = self.init()
        path = []
        actions = []
        solved = False

        for step_ind in range(self.episode_max_steps):
            path.append(state)
            action, obs, state, done = self.sample_next_node(obs, state)
            actions.append(action)
            if done:
                print("Solved")
                solved = True
                break

        print("Path len:", len(path))
        game = [(state.get_np_array_version(), 0.0, action) for action, state in zip(actions, path)]
        return game, solved

    def sample_next_node(self, obs, state):
        value = self._value([obs], [state])[0]

        assert len(value) == 5, "For sampling planner, `ValueBase.value_and_policy_net` must be set to True"
        policy = value[1:]
        temp_policy = self._apply_temperature(policy)
        chosen_action = np.random.choice(range(4), p=temp_policy)

        # neighbours are ordered in the order of actions: 0, 1, ..., _model.num_actions
        obs, rewards, dones, solved, states = self._model.neighbours(state)
        return chosen_action, obs[chosen_action], states[chosen_action], dones[chosen_action]

    def _apply_temperature(self, policy):
        logits = np.log(np.asarray(policy))  # + constant, which doesn't matter in temperature scaling
        return softmax(logits, self.temperature)
