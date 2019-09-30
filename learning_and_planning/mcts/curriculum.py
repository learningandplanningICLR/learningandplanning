
class Curriculum(object):
    def __init__(self,
                 enabled=False,
                 model=None,
                 curriculum_threshold=0.8,  # when WR exceeds this bump curriculum
                 curriculum_smooth_coeff=0.95,
                 curriculum_initial_value=0.0,  # initial value for avg WR
                 curriculum_initial_length_random_walk=50,  # starting 'number of steps' of agent moving box
                 curriculum_maximal_length_random_walk=300,  # max 'number of steps' of agent moving box
                 curriculum_length_random_walk_delta=50,  # the amount we bump the number of steps
                 curriculum_max_num_gen_steps=27,  # number of steps to generate level
                 curriculum_running_average_drop_on_increase=0.2,  # drop in avg WR after curriculum bump
                 ):

        self.enabled = enabled
        self.model = model
        self.threshold = curriculum_threshold
        self.smooth_coeff = curriculum_smooth_coeff
        self.length_random_walk = curriculum_initial_length_random_walk
        self.initial_length_random_walk = curriculum_initial_length_random_walk
        self.maximal_length_random_walk = curriculum_maximal_length_random_walk
        self.max_num_gen_steps = curriculum_max_num_gen_steps
        self.avg = curriculum_initial_value
        self.length_random_walk_delta = curriculum_length_random_walk_delta
        self.running_average_drop_on_increase = curriculum_running_average_drop_on_increase

    def apply(self, game_solved):
        ''' Performs curriculum bump in two stages:
            1) increases the number of steps the agent takes to set boxes aparat (random_walk)
            2) increases the size of room (number of the floor generating steps)
        '''
        if not self.enabled:
            return

        avg = self.smooth_coeff * self.avg + (1 - self.smooth_coeff) * game_solved
        if avg > self.threshold:  # do a curriculum bump
            if self.length_random_walk > self.maximal_length_random_walk:  # if max steps for box achieved bump floor generation steps
                if self.model.num_gen_steps < self.max_num_gen_steps:
                    self.length_random_walk = self.initial_length_random_walk
                    self.model.curriculum = self.length_random_walk
                    self.model.num_gen_steps += 1
            else:  # if box placement steps did not achieve maximum, bump it
                self.length_random_walk += self.length_random_walk_delta
                self.model.curriculum = self.length_random_walk
            avg -= self.running_average_drop_on_increase

    @property
    def num_gen_steps(self):
        return self.model.num_gen_steps

    @property
    def curriculum(self):
        return self.model.curriculum

