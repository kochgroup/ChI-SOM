"""
Utility tools for hyperparameter configuration for (emergent) SOMs
"""

from math import ceil, log, sqrt
from typing import Optional, Tuple

from chisom._core.utils import _decay_exponential, _decay_linear


def decay_linear(
    iteration: int,
    initial_value: int | float,
    total_iterations: Optional[int] = None,
    decay: Optional[float] = None,
    *args,
    **kwargs,
) -> float:
    """
    Calculate the linear decay of a value for use in the training process.
    Ether to use a fixed number of iterations or a decay factor. 'decay' takes
    precedence over 'total_iterations'.

    Parameters
    ----------
    iteration
        Current iteration.
    initial_value
        Starting value of the decay.
    total_iterations
        Total number of desired iterations, by default None.
    decay
        Decay rate, like m in y = -mx+b, by default None.

    Returns
    -------
    float
        Value at iteration `iteration`.

    Raises
    ------
    ValueError
        If neither `total_iterations` nor `decay` is provided.
    """
    if decay:
        return _decay_linear(iteration, initial_value, decay)
    elif total_iterations:
        decay = total_iterations
        return _decay_linear(iteration, initial_value, decay)
    else:
        raise ValueError("Either total_iterations or decay must be provided.")


def decay_exponential(
    iteration: int,
    initial_value: int | float,
    end_value: Optional[int | float] = None,
    total_iterations: Optional[int] = None,
    decay: Optional[float] = None,
    *args,
    **kwargs,
) -> float:
    """
    Calculate the exponential decay of a value towards a desire final value for use in the training process.
    Ether to use a fixed number of iterations or a decay factor. 'decay' takes precedence over 'total_iterations'.


    Parameters
    ----------
    iteration
        Current iteration.
    initial_value
        Starting value of the decay.
    end_value
        Desired final value, when not using decay by default None.
    total_iterations
        Total number of desired iterations, by default None.
    decay
        Decay rate, by default None.

    Returns
    -------
    float
        Value at iteration `iteration`.

    Raises
    ------
    ValueError
        If neither `total_iterations` nor `decay` is provided, or if both `end_value` and `decay` are provided.
    """
    if decay:
        return _decay_exponential(iteration, initial_value, decay)
    elif total_iterations and end_value:
        decay = total_iterations / -log((end_value) / initial_value)
        return _decay_exponential(iteration, initial_value, decay)
    else:
        raise ValueError(
            "Either total_iterations and end_value or decay must be provided."
        )


def lattice_size(dataset_size: int, factor=3) -> Tuple[int, int]:
    """
    Returns a rectangular lettice for the given number of data points,
    as recommendet by Ultsch et al. for ESOMs.

    Parameters
    ----------
    dataset_size
        Number of data points in the dataset.
    factor
        Ration of neurons to data points, by default 3.

    Returns
    -------
    Tuple[int, int]
        Number of (rows, columns)
    """
    n_neurons = dataset_size * factor
    rows = ceil(sqrt((n_neurons) / 1.5))
    columns = ceil(n_neurons / rows)
    return rows, columns
