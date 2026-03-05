import numpy as np


def _decay_exponential(iteration: int, init: float, decay: float) -> float:
    """
    This function implements an exponential decay function.

    Parameters
    ----------
    iteration : int
        The current iteration of the training loop.
    init : float
        The initial value of the parameter.
    decay : float
        The decay factor.

    Returns
    -------
    np.float32
        The decayed parameter value.

    """
    return float(init * np.exp(-iteration / decay))


def _decay_linear(iteration: int, init: float, decay: float) -> float:
    """
    This function implements a linear decay function.

    Parameters
    ----------
    iteration : int
        The current iteration of the training loop.
    init : float
        The initial value of the parameter.
    decay : float
        The decay factor.

    Returns
    -------
    np.float32
        The decayed parameter value.

    """
    return float(init - (init * (iteration / decay)))


def lazybatch(stop, batchsize, start=0):
    i = start
    while i < stop:
        next_i = i + min(batchsize, stop - i)
        to_yield = list(range(i, next_i))
        i = next_i
        yield to_yield
