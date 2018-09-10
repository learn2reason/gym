from __future__ import division
import numpy as np
from gym.envs.algorithmic import algorithmic_env
from gym.spaces import Box, Discrete

class SingleMultiplicationEnv(algorithmic_env.GridAlgorithmicEnv):
    def __init__(self, rows=1, base=3):
        self.singlen = 0
        super(SingleMultiplicationEnv, self).__init__(rows=rows, base=base, chars=False)
        self.observation_space = Box(low=0, high=base, shape=(2+1,))

    def target_from_input_data(self, input_strings):
        curry = 0
        target = []
        for digits in input_strings:
            total = digits[0]*self.singlen + curry
            target.append(total % self.base)
            curry = total // self.base

        if curry > 0:
            target.append(curry)

        newt = []
        start0 = True
        for t in list(reversed(target)):
            if t == 0 and start0:
                continue
            else:
                start0 = False
                newt.append(t)
        if newt == []:
            newt = [0]
        newt = list(reversed(newt))
        # print('newt', newt)
        return newt

    def generate_input_data(self, size):
        self.singlen = self.np_random.randint(self.base)
        return [
            [self.np_random.randint(self.base) for _ in range(self.rows)]
            for __ in range(size)
        ]

    def get_inputs(self):
        obs = self._get_obs()
        singlen = self.singlen
        r = [obs, singlen]
        return np.asarray(r)

    def reset(self):
        obs = super(SingleMultiplicationEnv, self).reset()
        self.read_head_position = (0, 0)
        return obs

    def render_observation(self):   
        x_str = super(SingleMultiplicationEnv, self).render_observation()
        label =      "Observation Grid    : "
        x_str += " *" + " " * (len(label))
        x_str += str(self.singlen)
        x_str += "\n"
        return x_str

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
