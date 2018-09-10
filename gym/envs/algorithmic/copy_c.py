"""
Task is to copy content from the input tape to
the output tape. http://arxiv.org/abs/1511.07275
"""
from gym.envs.algorithmic import algorithmic_env
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

class CopyEnv2(algorithmic_env.TapeAlgorithmicEnv):
    def __init__(self, base=5, chars=True):
        self.all_combs = []
        for numlens in range(1,2):
            y = get_combs(numlens, base)
            self.all_combs += y
        for numb in range(base):
            newc = []
            newc.append(numb)
            newc.append(0)
            self.all_combs.append(newc)
        self.index= 0
        super(CopyEnv2, self).__init__(base=base, chars=chars)

    def target_from_input_data(self, input_data):
        return input_data

    def generate_input_data(self, size):
        # return [self.np_random.randint(self.base) for _ in range(size)]
        r = self.all_combs[self.index]
        self.index += 1
        if self.index >= len(self.all_combs):
            self.index = 0
        return r

