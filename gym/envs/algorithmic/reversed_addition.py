from __future__ import division
import numpy as np
from gym.envs.algorithmic import algorithmic_env
from gym.spaces import Box, Discrete

class ReversedAdditionEnv(algorithmic_env.GridAlgorithmicEnv):
    def __init__(self, rows=2, base=3):
        super(ReversedAdditionEnv, self).__init__(rows=rows, base=base, chars=False)
        self.observation_space = Box(low=0, high=base, shape=(2+1,))

    def target_from_input_data(self, input_strings):
        curry = 0
        target = []
        for digits in input_strings:
            total = sum(digits) + curry
            target.append(total % self.base)
            curry = total // self.base

        if curry > 0:
            target.append(curry)
        return target

    def get_inputs(self):
        x = self.read_head_position
        r = []
        if x[1] == 1:
            r.append(self._get_obs((x[0], x[1])))
            r.append(self._get_obs((self.base, x[1])))
        else:
            r.append(self._get_obs((x[0], x[1])))
            r.append(self._get_obs((x[0],   x[1]+1)))
        return np.asarray(r)

    @property
    def time_limit(self):
        # Quirk preserved for the sake of consistency: add the length of the input
        # rather than the length of the desired output (which may differ if there's
        # an extra carried digit).
        # TODO: It seems like this time limit is so strict as to make Addition3-v0
        # unsolvable, since agents aren't even given enough time steps to look at
        # all the digits. (The solutions on the scoreboard seem to only work by
        # save-scumming.)
        return self.input_width*2 + 4
