import gin

@gin.configurable
class EnsembleConfigurator:

    def __init__(self, num_ensembles=1):
        self.num_ensembles = num_ensembles