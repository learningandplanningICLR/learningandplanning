from tensorflow.keras import backend as K

UNSOLVABLE_FLOAT = -9999.

def MSECut(yTrue, yPred):
  return K.mean(K.square(K.relu(K.abs(yTrue - yPred) - 0.25)))


def MSEsolvable(yTrue, yPred):
  solvable_mask = K.not_equal(yTrue, UNSOLVABLE_FLOAT)
  error_pointwise = K.square(yTrue - yPred)
  error_pointwise = error_pointwise * \
                    K.cast(solvable_mask, K.floatx())
  return K.mean(error_pointwise)


custom_losses = {
  "mse_cut": MSECut
}
