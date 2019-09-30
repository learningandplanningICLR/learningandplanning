import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
from tensorflow.python.keras import layers, Input

def keras_net(observation):
    """
      Simplified version (without dropout) of
      https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
      """

    # ob = Input(observation)
    net = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(observation)
    net = layers.Conv2D(64, (3, 3), activation='relu')(net)
    net = layers.MaxPooling2D(pool_size=(2, 2))(net)
    net = layers.Flatten()(net)
    net = layers.Dense(128, activation='relu')(net)

    logits = layers.Dense(4, activation=None)(net)
    vpred = layers.Dense(1, activation=None)(net)
    return logits, vpred



class KerasPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        logits, self.vpred = keras_net(ob)
        self.pd = pdtype.pdfromflat(logits)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample() # XXX
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []


