import gin
import numpy as np
import tensorflow as tf

@gin.configurable
class AutoMLCreator:

    def __init__(self,
                 auto_ml_mapping=None,
                 auto_ml_lr=0,
                 auto_ml_smoothing=0.995,
                 advantage_smoothing=0.999,
                 use_advantage=False,
                 starting_logits=None,
                 l2_loss_coeff=0.0):
        self.auto_ml_mapping = auto_ml_mapping
        self.auto_ml_lr = auto_ml_lr
        self.auto_ml_smoothing = auto_ml_smoothing
        self.advantage_smoothing = advantage_smoothing
        self.use_advantage = use_advantage
        self.starting_logits = starting_logits
        self.l2_loss_coeff = l2_loss_coeff
        if auto_ml_mapping is not None:

            self.dims = []
            for i, parameter_name in enumerate(self.auto_ml_mapping):
                 self.dims.append(len(self.auto_ml_mapping[parameter_name]))
        self.logs = None

    @property
    def is_auto_ml_present(self):
        return self.auto_ml_mapping is not None

    def fake_data(self):
        return np.zeros(shape=(len(self.dims,)))

    def data(self, parameters):
        return np.array(parameters)

    def auto_ml_dispatch_parameters(self, parameters):
        if not self.is_auto_ml_present:
            return {}

        assert len(parameters) == len(self.auto_ml_mapping), "Bad parameters"
        params_dict = {}
        for num, parameter_name in enumerate(self.auto_ml_mapping):
            parameter_idx = parameters[num]
            parameters_list = self.auto_ml_mapping[parameter_name]
            parameter_value = parameters_list[parameter_idx]
            params_dict[parameter_name] = parameter_value
        return params_dict

    def consume_logs(self, logs):
        self.logs = logs

    def print_logs(self, logger):
        estimators, solved_mean, policy, empirical_rewards = self.logs

        logger.record_tabular("auto_ml_solved_mean", solved_mean)
        # for i, estimator in enumerate(estimators):
        #     for j, e in enumerate(estimator):
        #         logger.record_tabular(f"auto_ml_estimator_{i}_{j}", e)
        for i, p in enumerate(policy):
            logger.record_tabular(f"auto_ml_prob_{i}", p)
        for i, rews in enumerate(empirical_rewards):
            for j, r in enumerate(rews):
                logger.record_tabular(f"auto_ml_adv_{i}_{j}", r)

    # PM: This is a helper method which shields bad design choices ;)
    @staticmethod
    def get_net():
        try:
            return tf.get_default_graph().get_tensor_by_name("model/auto_ml_net:0")
        except:
            pass

        try:
            return tf.get_default_graph().get_tensor_by_name("model_0/auto_ml_net:0")
        except:
            pass

        raise Exception("AutoML network not found")


@gin.configurable
def auto_ml_scalar_setter_1(value):
    return value
