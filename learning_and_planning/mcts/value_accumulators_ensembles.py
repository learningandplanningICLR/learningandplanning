import math
import random
from collections import deque
from itertools import combinations

import gin
import numpy as np

import sys

from learning_and_planning.mcts.ensemble_configurator import EnsembleConfigurator
from learning_and_planning.mcts.value_accumulators import ValueAccumulator


@gin.configurable
class EnsembleValueAccumulatorLogSumExp(ValueAccumulator):

    def __init__(self, value, state, kappa):
        self._sum = 0.0   # will work by broadcast
        self._sum_of_exps = 0.0   # will work by broadcast
        self._count = 0
        self._kappa = kappa
        super().__init__(value, state)
        # TODO: different index and target

    def add(self, value):
        self._sum += value  # value is a vector (from ensemble)
        self._sum_of_exps += np.exp(self._kappa * value)  # value is a vector (from ensemble)
        self._count += 1

    def add_auxiliary(self, value):
        return

    def get(self):
        if self._count:
            return self._sum / self._count
        else:
            return 0.0

    def get_sum_exps(self):
        if self._count:
            return self._sum_of_exps / self._count
        else:
            return 0.0

    def index(self, parent_value=None, action=None):
        return np.mean(self.get_sum_exps())

    def target(self):
        return np.mean(self.get())

    def count(self):
        return self._count

# Idea: 1/kappa * log \sum e(kappa* X) ~ E[X] + kappa/2 Var(X)
# Find kappa such that this is equal to UCB1, where UCB1 = sqrt(2log(n)/n)
# kappa = 2 * UCB1 / Var(X)
@gin.configurable
class EnsembleValueAccumulatorLogSumExp2(ValueAccumulator):

    def __init__(self, value, state, kappa, K=1, support=1.):
        # _sum = (s_1, ..., s_K), where s_k = x_1^k + ... + x_n^k
        self._sum = 0.0   # will work by broadcast
        self._sum_of_exps = 0.0   # will work by broadcast
        self.K = K  # number of ensembles
        self._count = 0

        self.support = support  # scales kappa, corresponds to (b-a) where P(X\in[a,b])=1
        self.epsilon = 1e4
        super().__init__(value, state)

    def kappa(self):
        n = self._count
        K = self.K
        variance_ensembles = np.var(self._sum)
        kappa = 2. * self.support * np.sqrt(2. * np.log(n) / n / K) / (variance_ensembles + self.epsilon)
        return kappa

    def add(self, value):
        self._sum += value  # value is a vector (from ensemble)
        self._sum_of_exps += np.exp(self.kappa() * value)  # value is a vector (from ensemble)
        self._count += 1

    def add_auxiliary(self, value):
        return

    def get(self):
        if self._count:
            return self._sum / self._count
        else:
            return 0.0

    def get_sum_exps(self):
        if self._count:
            return self._sum_of_exps / self._count
        else:
            return 0.0

    def index(self, parent_value=None, action=None):
        return np.log(np.mean(self.get_sum_exps())) / self.kappa()

    def target(self):
        return np.mean(self.get())

    def count(self):
        return self._count



# We have K independent bandits problem, each with different arm distribution, governed by theta
# We have no information about thetas, so assume uniform distribution on [1, ..., K]
# We assume that in each problem we use UCB1 (hence depends on n, not nK)
# We consider 1/kappa * log E_\theta e^(kappa (E[X|theta] + UCB1)), (E_\theta means E wrt to theta distribution)
# Which proxies to: E[X] + UCB1 + kappa/2 Var_theta(E[X|theta])
# (Note that Var_theta(E[X|theta] = Var(X) - E_theta[Var(X|theta)] <= Var(X))
# So on top of UCB1 we have an add-on on the level of disagreement between experiments
# How to choose kappa?
@gin.configurable
class EnsembleValueAccumulatorLogSumExp3(ValueAccumulator):

    def __init__(self, value, state, kappa_type="const"):  # [const, one_over_n, one_over_sqrt_n, sqrt_logn_n]
        # _sum = (s_1, ..., s_K), where s_k = x_1^k + ... + x_n^k
        self._sum = 0.0   # will work by broadcast
        self._sum_of_exps = 0.0   # will work by broadcast
        self._count = 0
        kappa_mapping = {"const": lambda n: 0.1,
                         "one_over_n": lambda n: 1. / n,
                         "one_over_sqrt_n": lambda n: 1. / np.sqrt(n),
                         "sqrt_logn_n": lambda n: np.sqrt(np.log(n)/n)
                         }
        self.kappa_fn = kappa_mapping[kappa_type]

        self.epsilon = 1e4
        super().__init__(value, state)

    def kappa(self):
        return self.kappa_fn(self._count)

    def ucb(self):
        return np.sqrt(2 * np.log(self._count) / self._count)

    def add(self, value):
        self._sum += value  # value is a vector (from ensemble)
        self._sum_of_exps += np.exp(self.kappa() * value)  # value is a vector (from ensemble)
        self._count += 1

    def add_auxiliary(self, value):
        return

    def get(self):
        if self._count:
            return self._sum / self._count
        else:
            return 0.0

    def get_sum_exps(self):
        if self._count:
            return self._sum_of_exps / self._count
        else:
            return 0.0

    def index(self, parent_value=None, action=None):
        return (np.log(np.mean(self.get_sum_exps())) + self.ucb()) / self.kappa()

    def target(self):
        return np.mean(self.get())

    def count(self):
        return self._count


class ConstKappa(object):
    def __init__(self, value):
        self.value = value

    def __call__(self, n):
        return self.value

    def __str__(self):
        return f"{self.value}"


@gin.configurable
class EnsembleValueAccumulatorMeanVarMaxUCB(ValueAccumulator):

    def __init__(self, value, state,
                 kappa_fn=lambda n: 0.1,
                 ucb_coeff=0.0,
                 max_coeff=0.0,
                 ensemble_to_use_indices=Ellipsis,
                 hash_to_num_fn=None,
                 noise_param=0.0):
        self._sum = 0.0   # will work by broadcast
        self._sum_of_exps = 0.0   # will work by broadcast
        self._count = 0
        self.number_of_ensembles = EnsembleConfigurator().num_ensembles
        if callable(kappa_fn):
            self.kappa_fn = kappa_fn
        else:
            self.kappa_fn = lambda n: kappa_fn
        self.ucb_coeff = ucb_coeff
        self.max_coeff = max_coeff
        self.ensemble_to_use_indices = ensemble_to_use_indices
        self.auxiliary_loss = 0.0
        if hash_to_num_fn is not None:
            ensemble_num = hash_to_num_fn(state._initial_state_hash, self.number_of_ensembles)
            self.ensemble_to_use_indices = slice(ensemble_num, ensemble_num+1)

        self.noise_param = noise_param
        super().__init__(value, state)

    @property
    def kappa(self):
        return self.kappa_fn(self._count)

    @property
    def ucb(self):
        return np.sqrt(2 * np.log(self._count) / self._count)

    def add_auxiliary(self, value):
        self.auxiliary_loss += value

    def add(self, value):
        self._sum += value  # value is a vector (from ensemble)
        self._count += 1

    def get(self):
        if self._count:
            return self._sum[self.ensemble_to_use_indices] / self._count
        else:
            return 0.0

    def index(self, parent_value=None, action=None):
        return np.mean(self.get()) + self.kappa*np.var(self.get()) + \
               self.max_coeff*np.max(self.get()) + self.ucb_coeff*self.ucb + self.auxiliary_loss + \
               self.noise_param * random.normalvariate(0, 1.0)

    def target(self):
        return np.mean(self.get())

    def count(self):
        return self._count


@gin.configurable
class EnsembleValueAccumulatorMeanStdMaxUCB(ValueAccumulator):

    def __init__(self, value, state,
                 kappa_fn=lambda n: 0.1,
                 kappa_target=None,
                 ucb_coeff=0.0,
                 max_coeff=0.0,
                 exploration_target=False
                 ):
        """

        Args:
            exploration_target: if use self.index() as a target (include bonuses from
                ucb, ensembles variance, etc. in target).
        """
        self._sum = 0.0   # will work by broadcast
        self._sum_of_exps = 0.0   # will work by broadcast
        self._count = 0
        self.kappa_fn = kappa_fn
        self.kappa_target = kappa_target
        self.ucb_coeff = ucb_coeff
        self.max_coeff = max_coeff
        # which ensemble members to use when calculating index
        self.index_indices = Ellipsis
        self.auxiliary_loss = 0.0
        self.exploration_target = exploration_target
        super().__init__(value, state)

    @property
    def kappa(self):
        return self.kappa_fn(self._count)

    @property
    def ucb(self):
        return -self._count

    def set_index_indices(self, index_indices):
        assert self.index_indices == Ellipsis, "this is meant to be used only once after __init__"
        self.index_indices = index_indices

    def ensemble_evaluation(self, index_indices=Ellipsis):
        vals = self.get(index_indices=index_indices)
        return np.mean(vals) + self.kappa * np.std(vals) + \
               self.max_coeff * np.max(vals)

    def add_auxiliary(self, value):
        self.auxiliary_loss += value

    def add(self, value):
        self._sum += value  # value is a vector (from ensemble)
        self._count += 1

    def get(self, index_indices=Ellipsis):
        if self._count:
            return self._sum[index_indices] / self._count
        else:
            return 0.0

    def index(self, parent_value=None, action=None):
        return self.ensemble_evaluation(index_indices=self.index_indices) + self.ucb_coeff * self.ucb + self.auxiliary_loss

    def final_index(self, parent_value=None, action=None):
        # index() without ucb bonus
        return self.ensemble_evaluation(index_indices=self.index_indices) + self.auxiliary_loss

    def target(self):
        # do NOT use self.index_indices here
        if self.exploration_target:
            return self.ensemble_evaluation()
        elif self.kappa_target:
            return np.mean(self.get()) + self.kappa_target * np.std(self.get())
        else:
            return np.mean(self.get())

    def count(self):
        return self._count



# Consider the following classical bandit problem (B, n, D),
# where B = number of bandits, n = length of episode, D = family of bandit payoffs distribution
# We assume that arm i has distribution D_i, where D_i(dx) = \int_\theta f_i(x|theta)p(theta)d theta
# and we assume that theta \in {theta_1, ..., theta_K}
# (i.e. there is a latent variable theta, which if known gives the distribution of x
# In this setup we could use the usual bandit algorithm, given that we can interact with bandits
# We consider the following variation of this problem:
# 1) We are not able to pull arms in the origina, (B, n, D), setup.
# 2) Instead we are able to simultanuously play K bandit problems, each of the form
#    (B, n, D_theta_1), ..., (B, n, D_theta_K).
# 3) Given the prior, we want to find the optimal arm in the original bandit problem.
# We want to exploit both the usual exploration strategy (think UCB, EXP, MOSS, PUCB, etc.) with
# the measure of how our estimates in the different K games differ.
# In particular we will use Hoeffding ineqality and Empirical version of Bernstein inequality, see
# Maurer, Pontil - Empirical Bernstein Bounds and Sample Variance Penalization, 2009 (Theorem 4)
# Gao, Zhou - On the Doubt about Margin Explanation of Boosting, 2012 (Theorem 6, better constant)
# Audibert, Munios, Szepesvari - Exploration-exploitation ... 2009 --> FCUK (UCB-V)
# More concretely, >> HERE BE FORMULAE <<
@gin.configurable
class EnsembleValueAccumulatorVariance(ValueAccumulator):

    def __init__(self, value, state, bonus_fn, bonus_loading=0.1):  # [const, one_over_n, one_over_sqrt_n, sqrt_logn_n]
        # _sum = (s_1, ..., s_K), where s_k = x_1^k + ... + x_n^k
        self._sum = 0.0   # will work by broadcast
        self._count = 0
        super().__init__(value)
        self.auxiliary_loss = 0.0
        self.bonus = bonus_fn
        self.bonus_loading = bonus_loading

    def add(self, value):
        self._sum += value  # value is a vector (from ensemble)
        self._count += 1

    def add_auxiliary(self, value):
        self.auxiliary_loss += value

    def get(self):
        if self._count:
            return self._sum / self._count
        else:
            return 0.0

    # INFO: index includes penalty for states on your path
    def index(self, parent_value=None, action=None):
        parent_count = parent_value.count()
        num_ensembles = len(self._sum)
        return self.target() + self.auxiliary_loss + self.bonus_loading * \
               self.bonus(self.get(), self.count(), parent_count, num_ensembles)

    def target(self):
        return np.mean(self.get())

    def count(self):
        return self._count


@gin.configurable
class EnsembleValueAccumulatorVoting(ValueAccumulator):

    def __init__(self, value, state, bonus_fn, bonus_loading=0.1):  # [const, one_over_n, one_over_sqrt_n, sqrt_logn_n]
        # _sum = (s_1, ..., s_K), where s_k = x_1^k + ... + x_n^k
        self._sum = 0.0   # will work by broadcast
        self._count = 0
        super().__init__(value)
        self.auxiliary_loss = 0.0
        self.bonus = bonus_fn
        self.bonus_loading = bonus_loading

    def add(self, value):
        self._sum += value  # value is a vector (from ensemble)
        self._count += 1

    def add_auxiliary(self, value):
        self.auxiliary_loss += value

    def get(self):
        if self._count:
            return self._sum / self._count
        else:
            return 0.0

    # INFO: index includes penalty for states on your path
    def index(self, parent_value=None, action=None):
        parent_count = parent_value.count()
        bonus = self.bonus(self.get(), self.count(), parent_count, 1)
        return np.array(self.get()) + self.bonus_loading * bonus + self.auxiliary_loss

    def target(self):
        return np.mean(self.get())

    def count(self):
        return self._count

    def final_index(self, parent_value=None, action=None):
        parent_count = parent_value.count()
        num_ensembles = len(self._sum)
        return self.target() + self.auxiliary_loss + self.bonus_loading * \
               self.bonus(self.get(), self.count(), parent_count, num_ensembles)

@gin.configurable
class EnsembleValueAccumulatorBayes(ValueAccumulator):

    def __init__(self, value, state, kappa, num_data=10):
        self._ensembles = deque(maxlen=num_data)
        self._weights = np.array([])
        self._kappa = kappa
        super().__init__(value, state)

    def add(self, value):
        self._ensembles.append(value)
        self._update_weights(value)

    def add_auxiliary(self, value):
        return

    def get(self):
        num_ensembles = len(self._ensembles[0])
        curr_num_data = len(self._ensembles)
        indices = np.random.randint(0, curr_num_data, num_ensembles)
        sample = np.array([
            self._ensembles[t][k] for k, t in enumerate(indices)
        ])
        return sample

    def index(self, parent_value=None, action=None):
        return logsumexp(np.array(self._ensembles), self._kappa, self._weights)

    def target(self):
        if not self._ensembles:
            return 0.0
        return np.mean(self._ensembles)

    def count(self):
        return len(self._ensembles)

    def _update_weights(self, new_ensemble):
        if len(self._weights) == 0:
            self._weights = np.array(
                [1/len(new_ensemble)] * len(new_ensemble)
            )  # weights vector
        data = np.array(self._ensembles)
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        sigma = np.maximum(sigma, 0.01)  # sigma can be 0 e.g. for small data
        new_weights = [
            self._density(e, m, s) for e, m, s in zip(new_ensemble, mu, sigma)
        ]
        self._weights = np.array([
            w * nw for w, nw in zip(self._weights, new_weights)
        ])
        self._weights /= np.sum(self._weights)

    # TODO: create more densities and self._density_type: e.g. normal, gumbel,
    # cauchy, laplace
    @staticmethod
    def _density(x, mu, sigma):
        return (
            1/(sigma * np.sqrt(2 * np.pi)) *
            np.exp(-(x - mu)**2 / (2 * sigma**2))
        )

def logsumexp(values, kappa, weights=None):
    def _functional(x):
        return np.exp(kappa * x)
    if len(values.shape) == 1:
        values = np.expand_dims(values, axis=0)
    if weights is None:
        weights = np.ones(values.shape[-1]) / values.shape[-1]
    ef_beta = np.mean(_functional(values), axis=0)
    index = np.sum([e * w for e, w in zip(ef_beta, weights)])
    index = np.log(index)
    return index

def std(values):
    return np.std(values)

def std_combinatorial(values):
    return np.sqrt(np.mean([(z1 - z2)**2 for z1, z2 in combinations(values, 2)])) # mean divides sum over (n choose 2)

# INFO: below is a PROXY to the actual empirical variance in Bernstein/Bennett inequality
# (from Gao, Zhou Theorem 6)
# (e.g. it is the variance of means, not the actual variance between individual X's)
# (this should not matter much, since we take max with UCB)
# FYI, second term: result += 7. * log2delta / (self._count * num_ensembles) / 3.
def bernstein(values, count, parent_count, num_ensembles):
    _std = std(values)
    log2delta = np.log(2.) + 4. * np.log(parent_count)
    return _std * np.sqrt(log2delta / count / num_ensembles)

def bernstein_combinatorial(values, count, parent_count, num_ensembles):
    _std = std_combinatorial(values)
    log2delta = np.log(2.) + 4. * np.log(parent_count)
    return _std * np.sqrt(log2delta / count / num_ensembles)

@gin.configurable
def ucb1(values, count, parent_count, num_ensembles):
    return np.sqrt(2. * np.log(parent_count) / count / num_ensembles)  # '/num_ensembles' is not a mistake

@gin.configurable
def ucb1_mul_std(values, count, parent_count, num_ensembles):
    return std(values) * ucb1(values, count, parent_count, num_ensembles)

@gin.configurable
def ucb1_std_combinatorial(values, count, parent_count, num_ensembles):
    return std_combinatorial(values) * ucb1(values, count, parent_count, num_ensembles)

def logp(x):
    return max(math.log(x), 0)

# https://hal.archives-ouvertes.fr/hal-01785705/document
@gin.configurable
def moss(values, count, parent_count, num_ensembles):
    return math.sqrt(logp(parent_count / (4. * count)) / (2. * count))
