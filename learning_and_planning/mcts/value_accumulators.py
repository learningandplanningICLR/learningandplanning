import math
from collections import deque

import gin
import numpy as np


class ValueAccumulator:

    def __init__(self, value, state=None):
        # Creates and initializes with typical add
        self.add(value)

    def add(self, value):
        """Adds an abstract value to the accumulator.

        Args:
            value: Abstract value to add.
        """
        raise NotImplementedError

    def add_auxiliary(self, value):
        """
        Additional value for traversals
        """
        raise NotImplementedError

    def get(self):
        """Returns the accumulated abstract value for backpropagation.

        May be non-deterministic.
        """
        raise NotImplementedError

    def index(self, parent_value, action):
        """Returns an index for selecting the best node."""
        raise NotImplementedError

    def target(self):
        """Returns a target for value function training."""
        raise NotImplementedError

    def count(self):
        """Returns the number of accumulated values."""
        raise NotImplementedError


@gin.configurable
class ScalarValueAccumulator(ValueAccumulator):

    def __init__(self, value, state=None, mean_max_coeff=1.0):
        self._sum = 0.0
        self._count = 0
        self._max = value
        self.mean_max_coeff = mean_max_coeff
        self.auxiliary_loss = 0.0
        super().__init__(value, state)

    def add(self, value):
        self._max = max(self._max, value)
        self._sum += value
        self._count += 1

    def add_auxiliary(self, value):
        self.auxiliary_loss += value

    def get(self):
        return (self._sum / self._count)*self.mean_max_coeff \
               + self._max*(1-self.mean_max_coeff)

    def index(self, parent_value=None, action=None):
        return self.get() + self.auxiliary_loss  # auxiliary_loss alters tree traversal in mcts

    def final_index(self, parent_value=None, action=None):
        return self.index(parent_value, action)

    def target(self):
        return self.get()

    def count(self):
        return self._count


@gin.configurable
class PolicyValueAccumulator(ValueAccumulator):

    # noinspection PyMissingConstructor
    def __init__(self, value_and_policy, state, policy_weight, method="pm0"):
        # assumes convention that value = value_and_policy[0]
        # and policy = value_and_policy[1:]
        self.policy_weight = policy_weight
        self.method = method

        if type(value_and_policy) is float:  # PM: special case of terminal node. Do it better.
            self.policy = object()
            self._sum = 0.0
        else:
            self.policy = value_and_policy[1:]
            self._sum = value_and_policy[0]
        self._count = 1

    def add(self, value):
        self._sum += value
        self._count += 1

    def add_auxiliary(self, value):
        return

    def get(self):
        if self._count:
            return self._sum / self._count
        else:
            return 0.0

    def index(self, parent_value, action):
        value_contribution = self.get()
        policy_contribution = parent_value.policy[action]
        if self.method == "pm0":
            ret = float(value_contribution + policy_contribution*(math.sqrt(self.policy_weight/self._count)))
        elif self.method == "alphazero":
            # similar to alphazero, only difference is that we aggregate counts
            # in graph not tree (in particular self._count might be larger than
            # parent_value._count)
            Q = value_contribution
            U = self.policy_weight * policy_contribution * np.sqrt(parent_value.count()) / (1 + self._count)
            ret = float(Q + U)
        elif self.method == "policy_then_value":
            # first use policy, then value
            if parent_value.count() < self.policy_weight:
                ret = policy_contribution
            else:
                ret = value_contribution
        elif self.method == "policy_when_sure":
            if (
                parent_value.count() < self.policy_weight and
                policy_contribution > 0.6
            ):
                ret = 10000. + value_contribution
            else:
                ret = value_contribution
        else:
            raise ValueError("unknown policy value mixing method {}".format(self.method))
        return ret

    def target(self):
        return self.get()

    def count(self):
        return self._count


#@gin.configurable
#class EnsembleValueAccumulatorLogSumExp(ValueAccumulator):
#
#    def __init__(self, value, kappa):
#        self._sum = 0.0   # will work by broadcast
#        self._sum_of_exps = 0.0   # will work by broadcast
#        self._count = 0
#        self._kappa = kappa
#        super().__init__(value)
#        # TODO: different index and target
#
#    def add(self, value):
#        self._sum += value  # value is a vector (from ensemble)
#        self._sum_of_exps += np.exp(self._kappa * value)  # value is a vector (from ensemble)
#        self._count += 1
#
#    def add_auxiliary(self, value):
#        return
#
#    def get(self):
#        if self._count:
#            return self._sum / self._count
#        else:
#            return 0.0
#
#    def get_sum_exps(self):
#        if self._count:
#            return self._sum_of_exps / self._count
#        else:
#            return 0.0
#
#    def index(self, parent_value=None, action=None):
#        return np.mean(self.get_sum_exps())
#
#    def target(self):
#        return np.mean(self.get())
#
#    def count(self):
#        return self._count

#@gin.configurable
#class EnsembleValueAccumulatorLogSumExp2(ValueAccumulator):
#
#    def __init__(self, value, kappa, K=1, support=1.):
#        # _sum = (s_1, ..., s_K), where s_k = x_1^k + ... + x_n^k
#        self._sum = 0.0   # will work by broadcast
#        self._sum_of_exps = 0.0   # will work by broadcast
#        self.K = K  # number of ensembles
#        self._count = 0
#
#        self.support = support  # scales kappa, corresponds to (b-a) where P(X\in[a,b])=1
#        self.epsilon = 1e4
#        super().__init__(value)
#
#    def kappa(self):
#        n = self._count
#        K = self.K
#        variance_ensembles = np.var(self._sum)
#        kappa = 2. * self.support * np.sqrt(2. * np.log(n) / n / K) / (variance_ensembles + self.epsilon)
#        return kappa
#
#    def add(self, value):
#        self._sum += value  # value is a vector (from ensemble)
#        self._sum_of_exps += np.exp(self.kappa() * value)  # value is a vector (from ensemble)
#        self._count += 1
#
#    def add_auxiliary(self, value):
#        return
#
#    def get(self):
#        if self._count:
#            return self._sum / self._count
#        else:
#            return 0.0
#
#    def get_sum_exps(self):
#        if self._count:
#            return self._sum_of_exps / self._count
#        else:
#            return 0.0
#
#    def index(self, parent_value=None, action=None):
#        return np.log(np.mean(self.get_sum_exps())) / self.kappa()
#
#    def target(self):
#        return np.mean(self.get())
#
#    def count(self):
#        return self._count
#
#
#@gin.configurable
#class EnsembleValueAccumulatorBayes(ValueAccumulator):
#
#    def __init__(self, value, kappa, num_data=10):
#        self._ensembles = deque(maxlen=num_data)
#        self._weights = np.array([])
#        self._kappa = kappa
#        super().__init__(value)
#
#    def add(self, value):
#        self._ensembles.append(value)
#        self._update_weights(value)
#
#    def add_auxiliary(self, value):
#        return
#
#    def get(self):
#        num_ensembles = len(self._ensembles[0])
#        curr_num_data = len(self._ensembles)
#        indices = np.random.randint(0, curr_num_data, num_ensembles)
#        sample = np.array([
#            self._ensembles[t][k] for k, t in enumerate(indices)
#        ])
#        return sample
#
#    def index(self, parent_value=None, action=None):
#        return logsumexp(np.array(self._ensembles), self._kappa, self._weights)
#
#    def target(self):
#        if not self._ensembles:
#            return 0.0
#        return np.mean(self._ensembles)
#
#    def count(self):
#        return len(self._ensembles)
#
#    def _update_weights(self, new_ensemble):
#        if len(self._weights) == 0:
#            self._weights = np.array(
#                [1/len(new_ensemble)] * len(new_ensemble)
#            )  # weights vector
#        data = np.array(self._ensembles)
#        mu = np.mean(data, axis=0)
#        sigma = np.std(data, axis=0)
#        sigma = np.maximum(sigma, 0.01)  # sigma can be 0 e.g. for small data
#        new_weights = [
#            self._density(e, m, s) for e, m, s in zip(new_ensemble, mu, sigma)
#        ]
#        self._weights = np.array([
#            w * nw for w, nw in zip(self._weights, new_weights)
#        ])
#        self._weights /= np.sum(self._weights)
#
#    # TODO: create more densities and self._density_type: e.g. normal, gumbel,
#    # cauchy, laplace
#    @staticmethod
#    def _density(x, mu, sigma):
#        return (
#            1/(sigma * np.sqrt(2 * np.pi)) *
#            np.exp(-(x - mu)**2 / (2 * sigma**2))
#        )
#
#
#def logsumexp(values, kappa, weights=None):
#    def _functional(x):
#        return np.exp(kappa * x)
#    if len(values.shape) == 1:
#        values = np.expand_dims(values, axis=0)
#    if weights is None:
#        weights = np.ones(values.shape[-1]) / values.shape[-1]
#    ef_beta = np.mean(_functional(values), axis=0)
#    index = np.sum([e * w for e, w in zip(ef_beta, weights)])
#    index = np.log(index)
#    return index

@gin.configurable
class PolicyEnsembleAccumulator(ValueAccumulator):

    # noinspection PyMissingConstructor
    def __init__(self, value_and_policy, state):
        # assumes convention that value = value_and_policy[:, 0]

        if len(value_and_policy.shape) == 1:  # PM: special case of terminal node. Do it better.
            self.policies = object()
            self._sum = 0.0
        else:
            self.policies = value_and_policy[:, 1:]
            self._sum = value_and_policy[:, 0]
        self._count = 1

    def add(self, value):
        self._sum += value[:, 0]
        self._count += 1

    def add_auxiliary(self, value):
        return

    def get(self):
        return 0.0

    def index(self, parent_value, action):
        return 0

    def target(self):
        return self.get()

    def count(self):
        return self._count
