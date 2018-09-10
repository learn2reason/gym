"""
Task is to return every nth character from the input tape.
http://arxiv.org/abs/1511.07275
"""
from __future__ import division
from gym.envs.algorithmic import algorithmic_env
from gym.spaces import Box, Discrete
import numpy as np

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

class DuplicatedInputEnv6(algorithmic_env.TapeAlgorithmicEnv):
    def __init__(self, duplication=2, base=5):
        self.index= 0
        all_combs = []
        self.all_combs = []
        for numlens in range(1,3):
            y = get_combs(numlens, base)
            all_combs += y
        for comb in all_combs:
            ncomb = []
            for c in comb:
                ncomb.append(c)
                ncomb.append(c)
            self.all_combs.append(ncomb)
        self.duplication = duplication
        super(DuplicatedInputEnv6, self).__init__(base=base, chars=True)
        self.observation_space = Discrete((base+1)*10)


    def generate_input_data(self, size):
        '''
        res = []
        if size < self.duplication:
            size = self.duplication
        for i in range(size//self.duplication):
            char = self.np_random.randint(self.base)
            for _ in range(self.duplication):
                res.append(char)
        return res
        '''
        r = self.all_combs[self.index]
        self.index += 1
        if self.index >= len(self.all_combs):
            self.index = 0
        return r

    def target_from_input_data(self, input_data):
        return [input_data[i] for i in range(0, len(input_data), self.duplication)]
