import attr


@attr.s
class DQNModel(object):
    q_argmax = attr.ib()
    net_outputs = attr.ib()
    replay_net_outputs = attr.ib()
    replay_next_target_net_outputs = attr.ib()


class ModelCreator(object):
    def build_networks(self, state_ph, _replay):
        raise NotImplementedError
