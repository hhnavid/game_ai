import numpy as np


def normalize(x, stats):
    """
    Normalizes x to have zero mean and unit varaince.
    :param x: vector to be normalized
    :param stats: A RunningMeanStd object
    :return: Normalized vector
    """
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def denormalize(x, stats):
    """
    denormalizes x using a running mean and std

    :param x: vector to be denormalized
    :param stats: (RunningMeanStd) the running mean and std of the input to normalize
    :return: Denormalized vector
    """
    if stats is None:
        return x
    return x * stats.std + stats.mean


class RunningMeanStdMPI(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # This implementation is taken from gym/baselines/common/mpi_running_mean_std.py::RunningMeanStd.
    def __init__(self, epsilon=1e-2, shape=()):

        self._sum = np.zeros(shape, 'float64')
        self._sumsq = np.ones(shape, 'float64') * epsilon
        self._count = np.ones((), 'float64') * epsilon
        self.shape = shape

        self.mean = np.zeros(shape, 'float64')
        self.std = np.ones(shape, 'float64') * epsilon

    def update(self, x):
        """
        Navid
        :param x: a (1,sDim) ndarray
        :return:
        """
        x = x.astype('float64')
        n = int(np.prod(self.shape))
        # Navid: totalvec contains {sum, sumsq, count} => it dim is n+n+1=2n+1
        totalvec = np.concatenate([x.sum(axis=0).ravel(), np.square(x).sum(axis=0).ravel(), np.array([len(x)],dtype='float64')])

        self._sum += totalvec[0:n].reshape(self.shape)      # Navid:sum of samples collected from all processes
        self._sumsq += totalvec[n:2*n].reshape(self.shape)  # Navid:sum of squared samples collected
        self._count += totalvec[2*n]                        # Navid:Num. of samples from which sum,sumsq is computed
        self.mean = self._sum / self._count
        self.std = np.sqrt(np.maximum(self._sumsq / self._count - np.square(self.mean), 1e-2))
