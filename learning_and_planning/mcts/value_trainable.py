import time

import numpy as np
import tensorflow as tf
import os
import gin
from functools import partial

from learning_and_planning.envs.sokoban_env_creator import get_callable
from learning_and_planning.mcts.auto_ml import AutoMLCreator
from learning_and_planning.mcts.ensemble_configurator import EnsembleConfigurator
from learning_and_planning.mcts.mpi_common import SERVER_RANK
from learning_and_planning.mcts.multi_tower_initializer import \
    create_multihead_initializers
from learning_and_planning.mcts.replay_buffer import circular_replay_buffer_mcts
from learning_and_planning.mcts.value import Value
from learning_and_planning.mcts.value_accumulators import ScalarValueAccumulator, PolicyValueAccumulator
from learning_and_planning.mcts.value_accumulators_ensembles import EnsembleValueAccumulatorLogSumExp, \
    EnsembleValueAccumulatorBayes, EnsembleValueAccumulatorLogSumExp2, EnsembleValueAccumulatorLogSumExp3
from baselines.common.models import get_network_builder
import baselines.common.tf_util as tf_util

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


# reshuffles batch which is structured [batch_size//2 positives, batch_size//2 negatives]
def reshuffle_batch(batch, num_split):
    batch_size = batch.shape[0]
    positive_examples = batch[:batch_size // 2]
    negative_examples = batch[batch_size // 2:]
    mini_batch_size = batch_size // num_split
    batch = [tf.concat([positive_examples[(i * mini_batch_size // 2):((i + 1) * mini_batch_size // 2)],
                        negative_examples[(i * mini_batch_size // 2):((i + 1) * mini_batch_size // 2)]],
                       axis=0)
             for i in range(num_split)]
    # batch = tf.concat(batch, axis=0)
    return batch


class NetworkTemplate(object):
    def __init__(self, model_fn, model_name):
        self._model_fn = model_fn
        # INFO: hack (requires model_fn using keras to have 'keras' in the name, e.g. model_fn_keras
        self._keras = 'keras' in model_name
        self._networks = {}

    def __call__(self, name):
        if name not in self._networks:
            if self._keras:
                with tf.name_scope(name):
                    self._networks[name] = self._model_fn()
            else:
                self._networks[name] = tf.make_template(name, self._model_fn)
        return self._networks[name]


@gin.configurable
class ValueBase(Value):

    broadcast_scope = "model"

    def __init__(self,
                 sess,
                 model_name,
                 traits,
                 obs_shape=None,  # if nenv=4, then obs_shape=(4+1, frame.shape)
                 activation="identity",  # "identity" or "sigmoid"
                 decay=0.001,  # coeff with l2 regularization loss
                 max_tf_checkpoints_to_keep=4,
                 replay: circular_replay_buffer_mcts.PoloWrappedReplayBuffer = None,
                 optimizer_fn=None,
                 learning_rate_fn=None,
                 loss_fn=None,
                 loss_clip_threshold=0.0,
                 label_smoothing_coeff=0.0,  # if 0.1, then 0 -> 0.1, 1 -> 0.9
                 two_headed=False,
                 separate_heads=False,
                 alive_weight=1.0,

                 value_and_policy_net=False,
                 policy_weight=0.1,
                 report_gradients=False,
                 alive_noise=0.0,
                 distance_weight_decay=None,
                 distance_asymmetry=1.0,
                 consistency_weight=0.0,
                 model_creator_template=None,
                 accumulator_fn=None,
                 action_space_dim=4,
                 ):
        assert obs_shape is not None, "State shape is None"
        self.learning_rate_ph = tf.placeholder(tf.float32, shape=(), name="learning_rate_ph")
        self._optimizer = optimizer_fn(learning_rate=self.learning_rate_ph)
        self.learning_rate_fn = learning_rate_fn
        self.report_gradients = report_gradients
        self.value_and_policy_net = value_and_policy_net
        self.action_space_dim = action_space_dim
        self.policy_weight = policy_weight

        if model_creator_template is None:

            model_builder = get_network_builder(model_name)

            if separate_heads:
                original_model_builder = model_builder
                def head(index, state):
                    with tf.variable_scope("head_{}".format(index)):
                        return original_model_builder(state, output_dim=1)
                def model_builder(state, output_dim=1):
                    return tf.concat([
                        head(index, state) for index in range(output_dim)
                    ], axis=-1)

            _model_name = model_builder.__name__
            if two_headed:
                model_builder = partial(model_builder, output_dim=2)
            if value_and_policy_net:
                model_builder = partial(model_builder, output_dim=1+self.action_space_dim)

            self._model_creator = NetworkTemplate(model_builder, _model_name)
        else:
            self._model_creator = model_creator_template

        self._replay = replay

        self._decay = decay
        self._obs_shape = obs_shape
        self.loss_fn = loss_fn
        self._loss_clip_threshold = loss_clip_threshold
        self._label_smoothing_coeff = label_smoothing_coeff
        self._two_headed = two_headed
        self._alive_weight = alive_weight
        self.train_steps = 0
        self._alive_noise = alive_noise
        self._activation_name = activation
        self._distance_weight_decay = distance_weight_decay
        self._distance_asymmetry = distance_asymmetry
        self._consistency_weight = consistency_weight
        self.accumulator_fn = accumulator_fn

        # activation refers to the function applied to logits of each ensemble
        if activation == "identity":
            self._activation = tf.identity
        elif activation == "sigmoid":
            raise NotImplementedError("PM: It seems that loss does not take activation into account. 15.07")
            # self._activation = tf.sigmoid
        elif activation == "exp_neg_relu":
            self._activation = lambda x: tf.exp(-tf.nn.relu(x))
        else:
            raise ValueError('Unrecognized activation {}.'.format(activation))

        self._observation_ph = tf.placeholder(tf.float32, shape=self._obs_shape, name="observation_ph")

        self._train_value, eval_value = self._build_graph()
        self._eval_value, self._eval_info = self._postprocess_eval_output(eval_value)
        self.losses_labels = []
        self._train_op = self._build_train_op()
        self._sess = sess
        self._saver = tf.train.Saver(max_to_keep=max_tf_checkpoints_to_keep)
        self.traits = traits
        self.auto_ml_create_train_op()

    def _build_graph(self):
        raise NotImplementedError

    def _build_train_op(self):
        raise NotImplementedError

    def _postprocess_eval_output(self, output):
        if self._two_headed:
            (alive_logits, distance_logits) = tf.unstack(output, axis=-1)
            # Soft mask. It works way better than a hard one.
            alive_mask = tf.sigmoid(alive_logits)
            # Works only with sparse value for now (value of a dead state is 0).
            distance = self._activation(distance_logits)
            value = alive_mask * distance
            return value, (alive_mask, distance)
        if self.value_and_policy_net:
            value = self._activation(output[..., 0])
            policy = tf.nn.softmax(output[..., 1:])
            value_policy = tf.concat([tf.expand_dims(value, axis=-1), policy], axis=-1)
            return value_policy, ()

        # else, standard only value :)
        return (self._activation(tf.squeeze(output, axis=-1)), ())


    # TODO: hardwired "model" can be inconvenient later on
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model")

    def get_variables_for_broadcast(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope=self.broadcast_scope)

    def _train_op_block(self, labels, logits, mask=None):
        clip_threshold = self._loss_clip_threshold
        # if self._loss == 'mse':
        #     loss_fn = tf.losses.mean_squared_error
        #     clip_threshold **= 2
        # elif self._loss == 'huber':
        #     loss_fn = tf.losses.huber_loss
        # elif self._loss == 'cross_entropy':
        #     loss_fn = tf.losses.sigmoid_cross_entropy  # tf.nn.sigmoid_cross_entropy_with_logits
        # else:
        #     raise ValueError('Unrecognized loss {}.'.format(self._loss))

        kwargs = {}
        if self._label_smoothing_coeff > 1e-3:
            kwargs['label_smoothing'] = self._label_smoothing_coeff

        losses = []
        all_logits = logits
        if self._two_headed:
            (alive_logits, distance_logits) = tf.unstack(logits, axis=-1)
            # Works only with sparse value for now (alive state iff label > 0).
            # This occurs before label smoothing.
            alive = tf.cast(labels > 0, tf.int64)
            # Train distance prediction only for alive states. This empirically
            # leads to better differentiation between alive states.
            distance_mask = tf.cast(alive, tf.float32)
            alive_logits += tf.random.normal(
                tf.shape(alive_logits), stddev=self._alive_noise
            )
            loss = tf.losses.sigmoid_cross_entropy(
                alive,
                alive_logits,
                **kwargs,
            ) * self._alive_weight
            loss_scale = 1.0 / max(
                self._alive_weight, self._consistency_weight, 1
            )
            losses.append(loss_scale * tf.reduce_mean(loss))
            distance_labels = labels
            self.losses_labels.append("two_head_logit")

            if self._consistency_weight:
                (logits_prev, logits_next) = all_logits
                (_, distance_prev) = tf.unstack(logits_prev, axis=-1)
                (_, distance_next) = tf.unstack(logits_next, axis=-1)
                loss = tf.losses.hinge_loss(
                    tf.ones_like(distance_logits),
                    tf.abs(distance_prev - distance_next) - 1,
                    reduction=tf.losses.Reduction.NONE,
                    weights=distance_mask,
                    **kwargs,
                ) * self._consistency_weight
                losses.append(loss_scale * loss)
                # distance_labels = labels
                self.losses_labels.append("consistency")

            if self._activation_name == "exp_neg_relu":
                # Change labels from exp(-distance) to just distance.
                labels = -tf.log(labels + 1e-6)
                if self._distance_weight_decay:
                    distance_mask *= tf.exp(
                        -labels / self._distance_weight_decay
                    )
                distance_mask *= tf.where(
                    distance_logits > labels,
                    tf.ones_like(labels) * self._distance_asymmetry,
                    tf.ones_like(labels),
                )
        elif self.value_and_policy_net:
            distance_labels, actions_labels = labels
            distance_logits = self._activation(logits[:, 0])
            distance_mask = tf.ones_like(distance_logits)
            actions_logits = logits[:, 1:]
            actions_labels_one_hot = tf.one_hot(actions_labels, self.action_space_dim)
            loss = tf.losses.softmax_cross_entropy(actions_labels_one_hot, actions_logits, **kwargs)
            losses.append(self.policy_weight * loss)
            self.losses_labels.append("policy")
            loss_scale = 1
        else:
            distance_logits = tf.squeeze(logits, axis=-1)
            distance_labels = labels
            distance_mask = mask
            if distance_mask is None:
                distance_mask = tf.ones_like(distance_logits)
            loss_scale = 1
        loss_ = self.loss_fn(
            distance_labels,  # labels
            distance_logits,  # predictions (mse/huber) or logits (cross_entropy)
            reduction=tf.losses.Reduction.NONE,
            weights=distance_mask
        )
        loss_ = tf.maximum(loss_ - clip_threshold, 0)
        loss_ = tf.reduce_mean(loss_)*loss_scale
        losses.append(loss_)
        self.losses_labels.append("value")
        params = self.get_trainable_variables()
        # Do not use l2 regularization for auto_ml as the training goes in different loop.
        params = [param for param in params if "auto_ml" not in param.name]
        loss_reg = 0.
        for p in params:
            loss_reg += tf.nn.l2_loss(p)
        loss_reg /= len(params)
        loss_reg = self._decay * loss_reg
        losses.append(loss_reg)
        self.losses_labels.append("l2_regularization")

        loss_total = sum(losses)
        if self.report_gradients:
            norms = []
            total_grads = []
            for loss in losses:
                grads = self._optimizer.compute_gradients(loss)
                gradients, _ = zip(*grads)
                norms.append(tf.global_norm(gradients))
                total_grads.extend(grads)
            optimize_op = self._optimizer.apply_gradients(total_grads)
        else:
            optimize_op = self._optimizer.minimize(loss_total)
            norms = []

        return optimize_op, loss_total, losses, norms

    def checkpoint(self, checkpoint_dir, iteration_number):
        if not tf.gfile.Exists(checkpoint_dir):
            return None
        # Call the Tensorflow saver to checkpoint the graph.
        self._saver.save(
            self._sess,
            os.path.join(checkpoint_dir, 'tf_ckpt'),
            global_step=iteration_number)

    def restore(self, checkpoint_dir, iteration_number):
        self._saver.restore(self._sess,
                            os.path.join(checkpoint_dir,
                                         'tf_ckpt-{}'.format(iteration_number)))
        return True

    def train_step(self):
        self.train_steps += 1
        if callable(self.learning_rate_fn):
            lr = self.learning_rate_fn(self.train_steps)
        else:
            lr = self.learning_rate_fn
        _, loss_total, losses, gradient_norms = self._sess.run(self._train_op, feed_dict={self.learning_rate_ph: lr})
        return loss_total, losses, gradient_norms, lr

    def moments(self, inputs):
        return 0, 0

    def initializers(self):
        return []

    def __call__(self, obs=None, states=None):
        val = self._sess.run(
            self._eval_value, feed_dict={self._observation_ph: obs})
        return val

    def additional_info(self, obs):
        return tuple(map(self.traits.distill_batch, self._sess.run(
            self._eval_info, feed_dict={self._observation_ph: obs}
        )))

    @property
    def additional_info_cmaps(self):
        if not self._two_headed:
            return ()
        else:
            return "Greens", "Purples"

    def create_accumulator(self, initial_value, state):
        return self.accumulator_fn(initial_value, state)

    def auto_ml_create_train_op(self):
        self.auto_ml_creator = AutoMLCreator()
        if not self.auto_ml_creator.is_auto_ml_present:
            return

        params = self.get_trainable_variables()
        # Do not use l2 regularization for auto_ml as the training goes in different loop.
        params = [param for param in params if "auto_ml" in param.name]
        loss_reg = 0.
        for p in params:
            loss_reg += tf.nn.l2_loss(p)
        loss_reg = self.auto_ml_creator.l2_loss_coeff * loss_reg

        auto_ml_net = self.auto_ml_creator.get_net()
        self.empirical_probability = [np.ones(shape=(dim_size,)) / dim_size for dim_size in self.auto_ml_creator.dims]
        self.empirical_adventage = [np.zeros(shape=(dim_size,)) for dim_size in self.auto_ml_creator.dims]

        num_ml_params = len(self.auto_ml_creator.auto_ml_mapping)
        # Reinforce
        self.auto_ml_policy = tf.nn.softmax(auto_ml_net)
        parameters_ph = tf.placeholder(shape=len(self.auto_ml_creator.dims), dtype=tf.int32)
        advantage_ph = tf.placeholder(shape=(), dtype=tf.float32)
        prob = tf.squeeze(tf.slice(self.auto_ml_policy, parameters_ph, [1]*num_ml_params))
        loss_pg = -tf.math.log(prob)*advantage_ph
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.auto_ml_creator.auto_ml_lr)

        loss_total = loss_pg + loss_reg
        optimize_op = optimizer.minimize(loss_total)
        self.auto_ml_optimize_op = optimize_op
        self.auto_ml_parameters_ph = parameters_ph
        self.auto_ml_advantage_ph = advantage_ph

        dist = tf.distributions.Categorical(probs=tf.reshape(self.auto_ml_policy, shape=[-1]))
        self.auto_ml_sample_op = dist.sample()
        self.is_solved_mean = 0.0

    def auto_ml_train(self, is_solved, parameters):
        parameters = [parameters]
        res = float(is_solved)
        if self.auto_ml_creator.use_advantage:
            res -= self.is_solved_mean

        self.is_solved_mean = self.auto_ml_creator.advantage_smoothing*self.is_solved_mean + \
                              (1.0 - self.auto_ml_creator.advantage_smoothing)*is_solved

        smoothing_param = self.auto_ml_creator.auto_ml_smoothing
        for i, p in enumerate(parameters):
            vec = np.zeros_like(self.empirical_probability[i])
            vec[p] = 1.0

            self.empirical_probability[i] = smoothing_param * self.empirical_probability[i] \
                                            + (1.0 - smoothing_param) * vec
            self.empirical_adventage[i][p] = smoothing_param * self.empirical_adventage[i][p] \
                                             + (1.0 - smoothing_param) * res

        _, policy = self._sess.run([self.auto_ml_optimize_op, self.auto_ml_policy],
                                   feed_dict={self.auto_ml_advantage_ph: res,
                                              self.auto_ml_parameters_ph: parameters})
        return self.empirical_probability, self.is_solved_mean, policy, self.empirical_adventage

    def auto_ml_sample(self):
        res = self._sess.run(self.auto_ml_sample_op)
        res_list = []
        for dim in self.auto_ml_creator.dims:
            res_list.append(res % dim)
            res = res // dim
        return res_list


@gin.configurable
class ValueVanilla(ValueBase):
    def __init__(self, **kwargs):
        super().__init__(traits=ScalarValueTraits(), **kwargs)

    def _build_graph(self):
        model_template = self._model_creator("model")  # shape = (batch, 1)
        replay_observations = tf.cast(self._replay.observations, tf.float32)
        if self._replay.num_steps:
            train_out = [
                model_template(step_obs)
                for step_obs in tf.unstack(replay_observations)
            ]
        else:
            train_out = model_template(replay_observations)
        return train_out, model_template(self._observation_ph)

    def _build_train_op(self):
        if self.value_and_policy_net:
            labels = self._replay.values, self._replay.actions
        else:
            labels = self._replay.values
            if self._replay.num_steps:
                labels = labels[0, ...]
        return self._train_op_block(labels=labels,  # shape = (batch, )
                                    logits=self._train_value)  # shape = (batch, )


@gin.configurable
class ValueEnsemble(ValueBase):

    broadcast_scope = "model|prior"

    def __init__(self,
                 num_ensemble,
                 include_prior=True,
                 prior_scale=1.,
                 accumulator_fn=None,
                 **kwargs):

        self._num_ensemble = num_ensemble
        self._prior_scale = prior_scale
        self._include_prior = include_prior
        traits = EnsembleValueTraits(num_ensemble)
        super().__init__(traits=traits, accumulator_fn=accumulator_fn, **kwargs)

    def _build_graph(self):
        batches = reshuffle_batch(self._replay.observations, self._num_ensemble)
        batches = [tf.cast(batch, tf.float32) for batch in batches]

        model_templates = [self._model_creator("model_{}".format(i))
                           for i in range(self._num_ensemble)]
        self._train_models = [model(batch)
                              for model, batch in zip(model_templates, batches)]
        self._eval_models = [model(self._observation_ph)
                             for model in model_templates]

        if self._include_prior:
            prior_templates = [self._model_creator("prior_{}".format(i))
                               for i in range(self._num_ensemble)]
            self._train_priors = [tf.stop_gradient(prior(batch))
                                  for prior, batch in zip(prior_templates, batches)]
            self._train_ensemble = [model + self._prior_scale * prior
                                    for model, prior in zip(self._train_models, self._train_priors)]
            self._eval_priors = [tf.stop_gradient(prior(self._observation_ph))
                                 for prior in prior_templates]
            eval_value = tf.stack([model + self._prior_scale * prior
                               for model, prior in zip(self._eval_models, self._eval_priors)], axis=1)
        else:
            self._train_ensemble = self._train_models
            eval_value = tf.stack(self._eval_models, axis=1)

        train_value = tf.concat(self._train_ensemble, axis=0)  # for training

        return train_value, eval_value

    def _build_train_op(self):
        if self.value_and_policy_net:
            labels = (
                tf.concat(reshuffle_batch(self._replay.values, self._num_ensemble), axis=0),
                tf.concat(reshuffle_batch(self._replay.actions, self._num_ensemble), axis=0)
            )
        else:
            labels = reshuffle_batch(self._replay.values, self._num_ensemble)
            labels = tf.concat(labels, axis=0)
        return self._train_op_block(labels=labels,
                                    logits=self._train_value)


@gin.configurable
class ValueEnsemble2(ValueBase):

    broadcast_scope = "model|prior"

    def __init__(self,
                 sess,
                 num_ensemble=None,
                 traits=None,
                 prior_scale=1.,
                 accumulator_fn=None,
                 checkpoint_paths=None,
                 **kwargs):

        self._num_ensemble = num_ensemble
        if self._num_ensemble is None:
            self._num_ensemble = EnsembleConfigurator().num_ensembles
        self._prior_scale = prior_scale
        self.checkpoint_paths = checkpoint_paths
        self.multihead_initializers = []
        if traits is None:
            traits = EnsembleValueTraits(self._num_ensemble)
        super().__init__(sess=sess, traits=traits, accumulator_fn=accumulator_fn, **kwargs)

    def initializers(self):
        return self.multihead_initializers

    def _build_graph(self):
        model_template = self._model_creator("model")
        prior_template = self._model_creator("prior")
        ob_ = tf.cast(self._replay.observations, tf.float32)
        train_value = model_template(ob_)
        if self._prior_scale is not None:
            train_prior = tf.stop_gradient(prior_template(ob_))
            train_value = train_value+self._prior_scale*train_prior

        self._train_value = train_value
        eval_value = model_template(self._observation_ph)
        if self._prior_scale is not None:
            eval_prior = tf.stop_gradient(prior_template(self._observation_ph))
            eval_value = eval_value+self._prior_scale*eval_prior
        eval_value = tf.expand_dims(eval_value, axis=-1)  # Quirk to be compatible with rest of the code

        if self.checkpoint_paths is not None:
            self.multihead_initializers = create_multihead_initializers(self.checkpoint_paths)

        return train_value, eval_value

    def _build_train_op(self):
        labels_batch = self._replay.values
        labels = tf.tensordot(labels_batch, [1.0]*self._num_ensemble, axes=0)
        # batch_size = self._replay.values.shape[0]
        # l = [x % self._num_ensemble for x in range(batch_size)]
        # indices = tf.constant(l, dtype=tf.uint8)
        # mask = tf.one_hot(indices, self._num_ensemble)
        mask = self._replay.masks
        labels_ = tf.reshape(labels, [-1])  # Quirk to be compatible with rest of the code
        train_value_ = tf.reshape(self._train_value, [-1])  # Quirk to be compatible with rest of the code
        train_value_ = tf.expand_dims(train_value_, axis=-1)  # Quirk to be compatible with rest of the code
        mask_ = tf.reshape(mask, [-1])
        return self._train_op_block(labels=labels_, logits=train_value_, mask=mask_)


@gin.configurable
class ValueConstant(Value):
    def __init__(self, value):
        self.value = value
        self.traits = ScalarValueTraits()

    def create_accumulator(self, initial_value, state):
        return ScalarValueAccumulator(initial_value, state)

    def train_step(self):
        return 0.

    def moments(self, inputs):
        return self.value, 0

    def __call__(self, obs, states=None):
        if obs is not None:
            size = len(obs)
        else:
            size = len(states)
        return np.array((self.value,) * size)

    def checkpoint(self, checkpoint_dir, iteration_number):
        pass


# always returns 0.
class ValueZero(ValueConstant):
    def __init__(self):
        super().__init__(value=0.)


@gin.configurable
class ValueRandom(Value):
    def __init__(self):
        self.traits = ScalarValueTraits()

    def create_accumulator(self, initial_value, state):
        return ScalarValueAccumulator(initial_value, state)

    def train_step(self):
        return 0.

    def __call__(self, obs, states=None):
        if obs is not None:
            size = len(obs)
        else:
            size = len(states)
        return np.random.random(size)

    def checkpoint(self, checkpoint_dir, iteration_number):
        pass


class MPIValueWrapper(object):

    def __init__(self, value):
        assert MPI, "MPI is not loaded"
        self.value = value
        self.comm = MPI.COMM_WORLD
        self.status = MPI.Status()
        self.rank = self.comm.Get_rank()
        var_list = value.get_variables_for_broadcast()
        self.setfromflat = tf_util.SetFromFlat(var_list)  # , dtype=var_list[0].dtype)
        self.getflat = tf_util.GetFlat(var_list)
        self.count_updates = 1
        self.shared_weights = None

    def _initialize_buffer(self):
        tmp_ = self.getflat()
        size = tmp_.shape[0]
        nbytes = tmp_.nbytes
        itemsize = int(nbytes/size)
        dtype = tmp_.dtype
        win = MPI.Win.Allocate_shared(nbytes, itemsize, comm = self.comm)
        buf, itemsize = win.Shared_query(0)
        self.shared_weights = np.ndarray(buffer=buf, dtype=dtype, shape=(size,))


    @classmethod
    def class_name(cls):
        return cls.__name__

    def sync(self):
        if self.shared_weights is None:
            self._initialize_buffer()
        self.push()
        self.comm.Barrier()
        self.update()

    def update(self):
        if self.rank != SERVER_RANK:
            self.setfromflat(self.shared_weights)

    def push(self):
        if self.rank == SERVER_RANK:
            weights = self.getflat()
            self.shared_weights[:] = weights # PM: important, we want to copy

    def __getattr__(self, name):
        if name in dir(self):
            func = getattr(self, name)
        else:
            func = getattr(self.value, name)
        return func

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    def __call__(self, *args, **kwargs):
        return self.value.__call__(*args, **kwargs)

    @property
    def unwrapped(self):
        return self.value
    

class ValueTraits:
    """Defines functions operating on abstract values.

    Abstract value is anything returned from the value function, i.e. a single
    number, an ensemble etc. Abstract values are vectors, i.e. can be added
    together and multiplied/divided by scalars.

    Attrs:
        zero: Abstract value of a terminal state.
        dead_end: Abstract value of a dead end.
    """

    zero = None
    dead_end = None

    def distill_batch(self, value_batch):
        """Distill a batch of abstract values into a batch of scalars.

        Args:
            value_batch: Batch of abstract values.

        Returns a batch of scalars.
        """
        raise NotImplementedError


@gin.configurable
class ScalarValueTraits(ValueTraits):
    zero = 0.0

    def __init__(self, dead_end_value=-1.0):
        self.dead_end = dead_end_value

    def distill_batch(self, value_batch):
        return np.reshape(value_batch, newshape=-1)


@gin.configurable
class EnsembleValueTraits(ValueTraits):

    def __init__(self, ensemble_size, dead_end_value=-1.0):
        self.zero = np.zeros(ensemble_size)
        self.dead_end = np.array([dead_end_value] * ensemble_size)

    def distill_batch(self, value_batch):
        return value_batch.mean(axis=1)
