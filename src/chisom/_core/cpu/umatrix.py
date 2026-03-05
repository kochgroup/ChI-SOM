from collections.abc import Callable

import numpy as np

from chisom._core.cpu.distance import make_universal_distance_func
from chisom._core.types import Codebook, UMatrix

FASTMATH_FLAG = False


def make_umatrix_calculation(vector_dist_norm: str) -> Callable[[Codebook], UMatrix]:
    """
    Create a function to calculate the U-Matrix for a given distance norm.

    Parameters
    ----------
    vector_dist_norm : str
        The type of vector distance norm to use for U-Matrix calculation.

    Returns
    -------
    Callable[[Codebook], MapValues]
        A function that takes a 3D array of shape (height, width, dimensions)
        and returns the U-Matrix as a 2D array of shape (height, width).
        The U-Matrix is normalized to the range [0, 1].
    """
    vector_dist_func = make_universal_distance_func(vector_dist_norm)

    def caculate_umatrix(codebook: Codebook) -> UMatrix:
        """
        Calculate the U-Matrix for a given 3D array of shape (height, width, dimensions).
        This is done by calculating the distances to the neighboring neurons
        and normalizing the resulting matrix to the range [0, 1].
        CAVE: This function assumes a toroidal topology

        Parameters
        ----------
        codebook : Codebook
            3D array of shape (height, width, dimensions) representing the data.

        Returns
        -------
        NDArray[np.float32]
            2D array of shape (height, width) representing the U-Matrix,
            normalized to the range [0, 1].
        """

        # TODO: Add option for non-toroidal topology
        indeces = codebook.shape[:2]
        umatrix = np.zeros(indeces, dtype=np.float16)

        # Calculate distances to the top neighboring neurons by shifting the array
        north_matrix = np.concat((codebook[-1:, :, :], codebook[:-1, :, :]), axis=0)
        north_distance = vector_dist_func(codebook, north_matrix)
        umatrix += north_distance

        # Use the previous distance to calculate the south neighbors, by shifting the distance matrix
        south_distance = np.concat(
            (north_distance[1:, :], north_distance[:1, :]), axis=0
        )
        umatrix += south_distance

        east_matrix = np.concat((codebook[:, -1:, :], codebook[:, :-1, :]), axis=1)
        east_distance = vector_dist_func(codebook, east_matrix)
        umatrix += east_distance

        west_distance = np.concat((east_distance[:, 1:], east_distance[:, :1]), axis=1)
        umatrix += west_distance

        # Min-max normalization of the U-Matrix
        u_min = np.min(umatrix)
        u_max = np.max(umatrix)
        umatrix = (umatrix - u_min) / (u_max - u_min)

        return umatrix

    return caculate_umatrix
