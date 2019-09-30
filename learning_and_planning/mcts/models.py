import gin
import tensorflow as tf
from tensorflow.python.layers.core import Dense

from baselines.common.models import register, mlp
import gin
import numpy as np
from learning_and_planning.mcts.auto_ml import AutoMLCreator
from learning_and_planning.mcts.ensemble_configurator import EnsembleConfigurator

slim = tf.contrib.slim


@register("convnet")
def convnet(observation, output_dim=1):
    net = tf.div(observation, 255.)
    net = slim.conv2d(net, 32, [8, 8], stride=4)
    net = slim.conv2d(net, 64, [4, 4], stride=2)
    net = slim.conv2d(net, 64, [3, 3], stride=1)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 512)
    output = slim.fully_connected(net, output_dim, activation_fn=None)
    return output


@gin.configurable
def convnet_mnist_output_scale(value=1.):
    return value

@register("convnet_mnist")
def convnet_mnist(observation, output_dim=1):
    auto_ml_creator = AutoMLCreator()
    if auto_ml_creator.is_auto_ml_present:
        init = np.zeros(shape=auto_ml_creator.dims, dtype=np.float32)
        if auto_ml_creator.starting_logits is not None:
            init = np.asanyarray(auto_ml_creator.starting_logits, dtype=np.float32)
        vars = tf.get_variable(name="auto_ml_values", initializer=tf.constant(init))
        tf.identity(vars, name="auto_ml_net")

    net = observation
    for _ in range(5):
        net = slim.conv2d(net, 64, [3, 3], stride=1, padding='SAME')
    # net = slim.max_pool2d(net, [2, 2], scope='pool2')
    # net = slim.conv2d(net, 64, [3, 3], stride=1, padding='SAME')
    # net = slim.max_pool2d(net, [2, 2])
    net = slim.flatten(net)
    net = slim.fully_connected(net, 128, activation_fn=tf.nn.relu)
    output = slim.fully_connected(net, output_dim, activation_fn=None)
    output = output * convnet_mnist_output_scale()
    return output


@register("convnet_mnist_multi_head")
@gin.configurable
def convnet_mnist_multi_head(observation, split_depth, output_dim=1):
    # 4 is for the last conv layer
    # 5 should not be used (just reshape)
    # 6 is for the two fully connected layers
    # 7 is for the last fully connected layer

    auto_ml_creator = AutoMLCreator()
    if auto_ml_creator.is_auto_ml_present:
        init = np.zeros(shape=auto_ml_creator.dims, dtype=np.float32)
        if auto_ml_creator.starting_logits is not None:
            init = np.asanyarray(auto_ml_creator.starting_logits, dtype=np.float32)
        vars = tf.get_variable(name="auto_ml_values", initializer=tf.constant(init))
        tf.identity(vars, name="auto_ml_net")

    assert output_dim == 1, "This parameter is kept for backward compatibility. Remove after deadline"
    assert split_depth < 8, "The split must be smaller than the number of layers"

    layers_fns = [lambda n: slim.conv2d(n, 64, [3, 3], stride=1, padding='SAME')]*5
    layers_fns.append(lambda n: slim.flatten(n))
    layers_fns.append(lambda n: slim.fully_connected(n, 128, activation_fn=tf.nn.relu))
    layers_fns.append(lambda n: slim.fully_connected(n, 1, activation_fn=None))

    net = [observation]
    for depth, layer_fn in enumerate(layers_fns):
        if depth < split_depth:
            net = [layer_fn(net[0])]
        else:
            len_ = len(net)
            net = [layer_fn(net[i % len_]) for i in range(EnsembleConfigurator().num_ensembles)]

    output = tf.concat(net, axis=-1)
    return output


@register("convnet_mnist_multi_towers")
@gin.configurable
def convnet_mnist_multi_towers(observation, tower_depth=5, output_dim=1):

    layers_fns = [lambda n: slim.conv2d(n, 64, [3, 3], stride=1, padding='SAME')]*tower_depth
    layers_fns.append(lambda n: slim.flatten(n))
    layers_fns.append(lambda n: slim.fully_connected(n, 128, activation_fn=tf.nn.relu))
    layers_fns.append(lambda n: slim.fully_connected(n, 1, activation_fn=None))

    nets = []
    for i in range(EnsembleConfigurator().num_ensembles):
        with tf.compat.v1.variable_scope(f"tower_{i}"):
            net = observation
            for depth, layer_fn in enumerate(layers_fns):
                net = layer_fn(net)

            nets.append(net)
    output = tf.concat(nets, axis=-1)
    return output

@register("convnet_mnist_multi_towers_learnable_ensemble")
@gin.configurable
def convnet_mnist_multi_towers_learnable_ensemble(observation, tower_depth=5, internal_ensemble=1,
                                                  output_dim=1, hidden_aggregator_layers=1,
                                                  last_layer_initializer=None):

    layers_fns = [lambda n: slim.conv2d(n, 64, [3, 3], stride=1, padding='SAME')]*tower_depth
    layers_fns.append(lambda n: slim.flatten(n))
    layers_fns.append(lambda n: slim.fully_connected(n, 128, activation_fn=tf.nn.relu))
    layers_fns.append(lambda n: slim.fully_connected(n, 1, activation_fn=None))

    assert EnsembleConfigurator().num_ensembles == 1, "The current code does not support != 1"

    nets = []
    for i in range(internal_ensemble):
        with tf.variable_scope(f"tower_{i}"):
            net = observation
            for depth, layer_fn in enumerate(layers_fns):
                net = layer_fn(net)

            nets.append(net)
    output = tf.concat(nets, axis=-1)
    output = tf.stop_gradient(output)

    if hidden_aggregator_layers == -1: # This is for sanity tests
        return output

    for _ in range(hidden_aggregator_layers):
        output = slim.fully_connected(output, 128, activation_fn=tf.nn.relu)

    if last_layer_initializer is not None:
        output = slim.fully_connected(output, 1, activation_fn=None,
                                      weights_initializer=tf.constant_initializer(last_layer_initializer))
    else:
        output = slim.fully_connected(output, 1, activation_fn=None)

    return output


@register("very_shallow_multi_head")
@gin.configurable
def very_shallow_multi_head(observation, split_depth, output_dim=1):

    layers_fns = [lambda n: slim.flatten(n), lambda n: slim.fully_connected(n, 1, activation_fn=None)]

    net = [observation]
    for depth, layer_fn in enumerate(layers_fns):
        if depth < split_depth:
            net = [layer_fn(net[0])]
        else:
            len_ = len(net)
            net = [layer_fn(net[i % len_]) for i in range(EnsembleConfigurator().num_ensembles)]

    output = tf.concat(net, axis=-1)
    return output


@register("convnet_mnist_two_towers")
def convnet_mnist_two_towers(observation, output_dim=5):
    assert output_dim == 5, "This is a hackish way of implementing separate value and policy net"
    output_value = convnet_mnist(observation, output_dim=1)
    output_policy = convnet_mnist(observation, output_dim=4)
    output = tf.concat([output_value, output_policy], axis=1)

    return output


@register("convnet_mnist_bottleneck")
def convnet_mnist_bottleneck(observation, output_dim=1):
    """
    Simplified version (without dropout) of
    https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    """
    net = observation
    for _ in range(5):
        net = slim.conv2d(net, 16, [1, 1], stride=1, padding='SAME')
        net = slim.conv2d(net, 64, [3, 3], stride=1, padding='SAME')
    net = slim.flatten(net)
    net = slim.fully_connected(net, 128, activation_fn=tf.nn.relu)
    output = slim.fully_connected(net, output_dim, activation_fn=None)
    return output

@register("linear")
@gin.configurable
def linear(observation, output_dim=1, output_rescale=1.):
    """Similiar to bsuite default agent network."""
    net = tf.layers.flatten(observation)
    net = Dense(output_dim, use_bias=False)(net)
    output = net * output_rescale
    return output


@register("linear_multi_head")
@gin.configurable
def linear_multi_head(observation, num_heads, split_depth=None, output_dim=1,
                      output_rescale=1.):
    """Similiar to bsuite default agent network."""
    del split_depth
    observation = tf.layers.flatten(observation)
    outputs = [
        Dense(output_dim, use_bias=False)(observation)* output_rescale
        for _ in range(num_heads)
    ]
    output = tf.concat(outputs, axis=-1)
    return output


@register("mlp_multi_head")
@gin.configurable
def mlp_multi_head(observation, num_heads, output_dim=1, output_rescale=1., num_hidden_layers=2):
    """Similar to bsuite default agent network."""
    net = observation
    net = mlp(num_layers=num_hidden_layers, num_hidden=50, activation=tf.nn.relu,
              layer_norm=False)(net)
    outputs = [
        Dense(output_dim, use_bias=True)(net) * output_rescale
        for _ in range(num_heads)
    ]
    output = tf.concat(outputs, axis=-1)
    return output


@register("multiple_mlps")
@gin.configurable
def multiple_mlps(observation, num_heads, output_dim=1, output_rescale=1., num_hidden_layers=2):
    """Similiar to bsuite default agent network."""
    outputs = list()
    ob = tf.contrib.layers.flatten(observation)

    for _ in range(num_heads):
        net = ob
        for _ in range(num_hidden_layers):
            net = Dense(50, activation=tf.nn.relu)(net)
        net = Dense(output_dim, use_bias=True)(net)
        net = net * output_rescale
        outputs.append(net)
    output = tf.concat(outputs, axis=-1)
    return output


@register("mlp_bsuite")
@gin.configurable
def mlp_bsuite(observation, output_dim=1, output_rescale=1.):
    """Similiar to bsuite default agent network."""
    net = observation
    net = mlp(num_layers=2, num_hidden=50, activation=tf.nn.relu,
              layer_norm=False)(net)
    output = slim.fully_connected(net, output_dim, activation_fn=None) * output_rescale
    return output


@register("mlp_sokoban")
def mlp_sokoban(observation, output_dim=1):
    net = observation
    net = mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False)(net)
    output = slim.fully_connected(net, output_dim, activation_fn=None)
    return output
