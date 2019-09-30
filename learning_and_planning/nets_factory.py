import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import regularizers, layers
from tensorflow.keras.layers import Dropout, BatchNormalization
from baselines.common.models import register


def get_network(name, **parameters):
  if name == "pm_conv_net_value":
    return pm_conv_net_value(**parameters)
  elif name == "kc_residual_conv_net_value":
    return kc_residual_conv_net_value(**parameters)
  elif name == "kc_small_mnist_cnn":
    return kc_small_mnist_cnn(**parameters)
  elif name == "kc_multihead_cnn":
    return kc_multihead_cnn(**parameters)
  elif name == "kc_parametrized_cnn_v0_2":
    return kc_parametrized_cnn_v0_2(**parameters)
  elif name == "kc_parametrized_resnet_v0_0":
    return kc_parametrized_resnet_v0_0(**parameters)
  raise NotImplementedError("network name unknown {}".format(name))

def pm_conv_net_value(conv_outs, conv_shapes, conf_strides,
                      conv_dropout, fc_dropout, loss_clip, fc_outs, conv_l2=0,
                      fc_l2=0):

  net = tf.keras.Sequential()


  for filters, kernel_size, stride in zip(conv_outs, conv_shapes, conf_strides):
    net.add(layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=stride,
        activation='relu', kernel_regularizer=regularizers.l2(conv_l2),))
    net.add(layers.Dropout(conv_dropout))

  net.add(layers.Flatten())

  for fc_out in fc_outs:
    net.add(layers.Dense(units=fc_out, activation='relu',
                         kernel_regularizer=regularizers.l2(fc_l2)),)
    net.add(layers.Dropout(fc_dropout))

  net.add(layers.Dense(units=1))

  return net


def kc_residual_conv_net_value(conv_outs, conv_shapes, conv_l2, n_blocks,
                               final_activation, input_shape=(8,8,7),
                               output_size=1, target_bias_initialization=0.):
  """

  Args:
    target_bias_initialization - initial values for bias in finall  layer.
      (by default equal to 0 in keras)
  """
  # This is fixed for now
  input = layers.Input(shape=input_shape)

  def residual_block(y):
    shortcut = y
    for filters, kernel_size in zip(conv_outs, conv_shapes):
      y = layers.Conv2D(
          filters, kernel_size=kernel_size, strides=(1, 1), activation="relu",
          padding='same', kernel_regularizer=regularizers.l2(conv_l2))(y)
    y = layers.add([shortcut, y])
    return y

  x = input
  for _ in range(n_blocks):
    x = residual_block(x)

  x = layers.GlobalAveragePooling2D()(x)

  if not target_bias_initialization:
    target_bias_initialization = 0.

  pred = layers.Dense(
      units=output_size, activation=final_activation,
      bias_initializer=tf.constant_initializer(
          target_bias_initialization, dtype=tf.float32
      )
  )(x)

  model = keras.models.Model(inputs=input, outputs=pred)

  return model


def kc_small_mnist_cnn(final_activation, input_shape=(8,8,7),
                       output_size=1):
  """

  Simplified version (without dropout) of
  https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
  """
  model = tf.keras.Sequential()
  model.add(layers.Conv2D(32, kernel_size=(3, 3),
                            activation='relu', input_shape=input_shape))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  # model.add(Dropout(0.25))
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))
  # model.add(Dropout(0.5))
  model.add(layers.Dense(output_size, activation=final_activation))

  return model


def kc_multihead_cnn(final_activation, input_shape=(8,8,7), output_size=1):
  """

  Simplified version (without dropout) of
  https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
  """

  input = layers.Input(shape=input_shape)

  x = input

  x = layers.Conv2D(32, kernel_size=(3, 3),
                    activation='relu', input_shape=input_shape)(x)

  x = layers.Conv2D(64, (3, 3), activation='relu')(x)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)
  # model.add(Dropout(0.25))
  x = layers.Flatten()(x)
  x = layers.Dense(128, activation='relu')(x)
  # model.add(Dropout(0.5))
  if isinstance(output_size, dict):
    pred = [
        layers.Dense(output_size[name],
                     activation=final_activation[name],
                     name=name
                     )(x)
      for name in output_size.keys()
    ]
  else:
    pred = layers.Dense(output_size, activation=final_activation)(x)

  model = keras.models.Model(inputs=input, outputs=pred)

  return model


def cnn_body_v0_2(x, l2=0, channels=64, n_layers=2, final_pool_size=(2,2),
                  batch_norm=False):
  """

  Args:
    x: tensor
  """

  for _ in range(n_layers):
    x = layers.Conv2D(channels, kernel_size=(3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(l2),
                      activation='relu',
                      )(x)
    if batch_norm:
      x = BatchNormalization()(x)
  x = layers.MaxPooling2D(pool_size=final_pool_size)(x)
  return x


def flatten_and_mlp_v0_1(
        x, n_hidden=128, n_layers=1, dropout=0, dropout_input=False, l2=0.):
  """
    Args:
      x: tensor
  """
  if dropout and dropout_input:
    x = Dropout(dropout)(x)
  x = layers.Flatten()(x)
  for _ in range(n_layers):
    x = layers.Dense(n_hidden, activation='relu',
                     kernel_regularizer=regularizers.l2(l2))(x)
    if dropout:
      x = Dropout(dropout)(x)
  return x



def net_head(x, final_activation, output_size):
  if isinstance(output_size, dict):
    pred = [
        layers.Dense(output_size[name],
                     activation=final_activation[name],
                     name=name
                     )(x)
      for name in output_size.keys()
    ]
  else:
    pred = layers.Dense(output_size, activation=final_activation)(x)
  return pred


@register("kc_parametrized_cnn_v0_2")
def kc_parametrized_cnn_v0_2_keras(final_activation=None, input_shape=(8, 8, 7),
                             output_size=1, cnn_l2=0, cnn_channels=64,
                             cnn_n_layers=2, cnn_final_pool_size=(2,2),
                             cnn_batch_norm=False,
                             global_average_pooling=False,
                             fc_n_hidden=128, fc_n_layers=1,
                             fc_dropout=0, fc_dropout_input=False, fc_l2=0.):
  """ Simple convolutional architecture.

  Args:
    global_average_pooling: if use GAP after convolutions, note that fully
      connected layers still will be applied (if fc_n_layers > 0). Setting True
      enables to infer network on different input shapes.
  """
  if global_average_pooling:
    # Remove height and width to enable inference on different image shapes.
    input_shape = (None, None, input_shape[2])
  input = layers.Input(shape=input_shape)
  x = cnn_body_v0_2(input, l2=cnn_l2, channels=cnn_channels,
                    n_layers=cnn_n_layers, final_pool_size=cnn_final_pool_size,
                    batch_norm=cnn_batch_norm)
  if global_average_pooling:
    x = layers.GlobalAveragePooling2D()(x)
  x = flatten_and_mlp_v0_1(x, n_hidden=fc_n_hidden, n_layers=fc_n_layers,
                           dropout=fc_dropout, dropout_input=fc_dropout_input,
                           l2=fc_l2)
  pred = net_head(x, final_activation=final_activation, output_size=output_size)
  model = keras.models.Model(inputs=input, outputs=pred)
  return model


@register("shallow_cnn_10_10")
def shallow_cnn_10_10_keras():
  return kc_parametrized_cnn_v0_2_keras(input_shape=(10, 10, 7),
                                        output_size=1)


def kc_parametrized_resnet_v0_0(
        final_activation, input_shape=(8, 8, 7), output_size=1,
        cnn_l2=0, cnn_channels=64, cnn_n_layers=2, cnn_final_pool_size=(1, 1),
        cnn_batch_norm=False, fc_n_hidden=128, fc_n_layers=1, fc_dropout=0,
        fc_l2=0., fc_dropout_input=False, n_residual_blocks=2,

):
  """

  Simplified version (without dropout) of
  https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
  """
  assert cnn_final_pool_size == (1, 1)
  net_input = layers.Input(shape=input_shape)
  x = net_input

  # Initial transformation of input.
  x = cnn_body_v0_2(
    x, l2=cnn_l2, channels=cnn_channels, n_layers=1,
    final_pool_size=cnn_final_pool_size, batch_norm=cnn_batch_norm,
  )

  # Residual blocks.
  for _ in range(n_residual_blocks):
    previous_x = x
    x = cnn_body_v0_2(
      x, l2=cnn_l2, channels=cnn_channels, n_layers=cnn_n_layers,
      final_pool_size=cnn_final_pool_size, batch_norm=cnn_batch_norm,
    )
    x = layers.add([previous_x, x])

  x = flatten_and_mlp_v0_1(x, n_hidden=fc_n_hidden, n_layers=fc_n_layers,
                           dropout=fc_dropout, dropout_input=fc_dropout_input,
                           l2=fc_l2)

  pred = net_head(x, final_activation=final_activation, output_size=output_size)

  model = keras.models.Model(inputs=net_input, outputs=pred)

  return model
