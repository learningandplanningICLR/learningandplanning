from tensorflow.python import keras
from tensorflow.python.keras import regularizers, layers
from tensorflow.python.keras.layers import Dropout, BatchNormalization

from learning_and_planning.supervised.supervised_target import Target


def get_network(name, **parameters):
  if name == "next_frame_and_done_cnn":
    return next_frame_and_done_cnn(**parameters)
  raise NotImplementedError("network name unknown {}".format(name))


def cnn_body_v0_2(x, l2=0, channels=64, n_layers=2, final_pool_size=(2,2),
                  batch_norm=False, strides=(1, 1), kernel_size=(3, 3),
                  dilation=(1, 1)):
  """

  Args:
    x: tensor
  """

  for _ in range(n_layers):
    x = layers.Conv2D(channels, kernel_size=kernel_size, strides=strides,
                      padding="same", kernel_regularizer=regularizers.l2(l2),
                      activation='relu', dilation_rate=dilation,
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


def next_frame_and_done_cnn(final_activation, input_shape=(8, 8, 7),
                             output_size=1, cnn_l2=0, cnn_channels=64,
                             cnn_n_layers=2, cnn_final_pool_size=(2,2),
                             cnn_batch_norm=False, cnn_kernel_size=(3,3),
                             cnn_strides=(1,1), cnn_dilation=(1,1),
                             global_average_pooling=False,
                             fc_n_hidden=128, fc_n_layers=1,
                             fc_dropout=0, fc_dropout_input=False, fc_l2=0.,
                             image_output=False):
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
                    batch_norm=cnn_batch_norm, strides=cnn_strides,
                    kernel_size=cnn_kernel_size, dilation=cnn_dilation)

  x_img = x

  if global_average_pooling:
    x = layers.GlobalAveragePooling2D()(x)


  x = flatten_and_mlp_v0_1(x, n_hidden=fc_n_hidden, n_layers=fc_n_layers,
                           dropout=fc_dropout, dropout_input=fc_dropout_input,
                           l2=fc_l2)

  # apply final transformations
  # x_img -> next_frame (logits map)
  # x -> if_done (logit)
  x_and_name = [
    (x_img, Target.NEXT_FRAME.value),
    (x, "if_done"),
  ]
  pred = [
    layers.Dense(
      output_size[name], activation=final_activation[name], name=name
    )(activation)
      for activation, name in x_and_name
    ]
  print("TODO remove")

  model = keras.models.Model(inputs=input, outputs=pred)
  return model
