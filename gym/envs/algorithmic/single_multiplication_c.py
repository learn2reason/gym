from __future__ import division
import numpy as np
from gym.envs.algorithmic import algorithmic_env
from gym.spaces import Box, Discrete

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def get_combs(numlens, base=5):
        x1 = np.arange(base)
        xs = []
        for i in range(numlens):
            xs.append(np.copy(x1))
        y = cartesian(xs)
        y = y.astype(int)
        return y.tolist()

class SingleMultiplicationEnv2(algorithmic_env.GridAlgorithmicEnv):
    def __init__(self, rows=1, base=3):
        self.singlen = 0
        f_combs = []
        for c in range(base):
            f_combs.append([[c]])
        for c in range(base):
            newc = []
            newc.append([2])
            newc.append([c])
            f_combs.append(newc)
        for c in range(base):
            newc = []
            newc.append([2])
            newc.append([2])
            newc.append([c])
            f_combs.append(newc)
            newc = []
            newc.append([2])
            newc.append([1])
            newc.append([c])
            f_combs.append(newc)
        self.all_combs = f_combs

        self.sn_combs = range(base)

        self.index1= 0
        self.index2= 0
        super(SingleMultiplicationEnv2, self).__init__(rows=rows, base=base, chars=False)
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
        # self.singlen = self.np_random.randint(self.base)
        # return [
        #     [self.np_random.randint(self.base) for _ in range(self.rows)]
        #     for __ in range(size)
        # ]
        r = self.all_combs[self.index1][:]
        self.singlen = self.sn_combs[self.index2]
        self.index1 += 1
        if self.index1 >= len(self.all_combs):
            self.index1 = 0
            self.index2 += 1
            if self.index2 >= len(self.sn_combs):
                self.index2 = 0
        return r

    def get_inputs(self):
        obs = self._get_obs()
        singlen = self.singlen
        r = [obs, singlen]
        return np.asarray(r)

    def reset(self):
        obs = super(SingleMultiplicationEnv2, self).reset()
        self.read_head_position = (0, 0)
        return obs

    def render_observation(self):   
        x_str = super(SingleMultiplicationEnv2, self).render_observation()
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
