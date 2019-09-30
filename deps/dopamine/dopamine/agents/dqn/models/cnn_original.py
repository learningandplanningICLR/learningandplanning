import collections

import tensorflow as tf

from dopamine.agents.dqn.models.utils import DQNModel, ModelCreator

slim = tf.contrib.slim

class CNNOriginal(ModelCreator):
    def __init__(self, num_actions):
        self.num_actions = num_actions


    def _get_network_type(self):
        """Returns the type of the outputs of a Q value network.

    Returns:
      net_type: _network_type object defining the outputs of the network.
    """
        return collections.namedtuple('DQN_network', ['q_values'])

    def _network_template(self, state):
        """Builds the convolutional network used to compute the agent's Q-values.

    Args:
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
        net = tf.cast(state, tf.float32)
        net = tf.div(net, 255.)
        net = slim.conv2d(net, 32, [8, 8], stride=4)
        net = slim.conv2d(net, 64, [4, 4], stride=2)
        net = slim.conv2d(net, 64, [3, 3], stride=1)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 512)
        q_values = slim.fully_connected(net, self.num_actions, activation_fn=None)
        return self._get_network_type()(q_values)

    def build_networks(self, state_ph, _replay):
        """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """
        # Calling online_convnet will generate a new graph as defined in
        # self._get_network_template using whatever input is passed, but will always
        # share the same weights.
        online_convnet = tf.make_template('Online', self._network_template)
        target_convnet = tf.make_template('Target', self._network_template)
        net_outputs = online_convnet(state_ph)
        # TODO(bellemare): Ties should be broken. They are unlikely to happen when
        # using a deep network, but may affect performance with a linear
        # approximation scheme.
        q_argmax = tf.argmax(net_outputs.q_values, axis=1)[0]

        replay_net_outputs = online_convnet(_replay.states)
        replay_next_target_net_outputs = target_convnet(_replay.next_states)
        return DQNModel(q_argmax=q_argmax,
                        replay_net_outputs=replay_net_outputs,
                        replay_next_target_net_outputs=replay_next_target_net_outputs)
