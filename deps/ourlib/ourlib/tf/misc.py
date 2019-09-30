import tensorflow as tf


def describe_tf_variable(sess, v):
    mean, variance = sess.run([tf.reduce_mean(v),
                               tf.reduce_mean((v - tf.reduce_mean(v)) ** 2)])

    print('name = {}, mean = {}, var = {}'.format(v.name, mean, variance))


def show_tensorflow_variables(sess):
    for v in tf.all_variables():
        describe_tf_variable(sess, v)


def restore_from_checkpoint_path(sess, path):
    print('Restoring from checkpoint {}'.format(path))
    saver = tf.train.Saver()
    saver.restore(sess, path)
