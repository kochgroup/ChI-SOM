import numpy as np
from numba import jit, njit, pndindex, prange

from chisom._core.cpu.types import (
    CodebookCoefficientsFunction,
    CodebookUpdateFunction,
    LocalUpdateFunction,
    Neighbors,
)
from chisom._core.types import Codebook, MapValues, Position, Vector

FASTMATH_FLAG = False


def make_kernel_coefficient_function(
    kernel: str, fastmath_flag: bool = FASTMATH_FLAG
) -> CodebookCoefficientsFunction:
    """
    Creates a kernel coefficient function for the specified kernel type.

    Parameters
    ----------
    kernel : str
        The type of kernel to create the coefficient function for. Supported kernels are "gaussian", "mexican", and "cone".
    fastmath_flag : bool, optional
        Using reduced precision for faster calculation, by default FASTMATH_FLAG

    Returns
    -------
    CodebookCoefficientsFunction
        A function that computes the coefficients for the specified kernel type.

    Raises
    ------
    ValueError
        If the specified kernel type is not supported.
    """
    if kernel == "gaussian":

        @njit(
            [
                "float32[:,::1](float32[:,::1], float32, float32)",
            ],
            fastmath=fastmath_flag,
            parallel=True,
        )
        def gaussian_coeff(
            map_distances: MapValues,
            alpha: np.float32,
            sigma: np.float32,
        ) -> MapValues:
            """
            Computes the gaussian coefficients for the update of the codebook. The coefficients are computed for all neurons in the codebook with respect to the BMU

            Parameters
            ----------
            map_distances : MapValues
                Map distances of all neurons in the codebook with respect to the BMU
            alpha : np.float32
                Current learning rate
            sigma : np.float32
                Current neighborhood size

            Returns
            -------
            MapValues
                Gaussian coefficients for all neurons in the codebook
            """
            rows, columns = map_distances.shape
            coefficients = np.zeros((rows, columns), dtype=np.float32)
            for neuron_index in pndindex((rows, columns)):
                coefficients[neuron_index] = alpha * np.exp(
                    -((map_distances[neuron_index] ** 2) / (sigma**2))
                )
            return coefficients

        return gaussian_coeff

    elif kernel == "mexican":

        @njit(
            [
                "float32[:,::1](float32[:,::1], float32, float32)",
            ],
            fastmath=fastmath_flag,
            parallel=True,
        )
        def mexican_coeff(
            map_distances: MapValues,
            alpha: np.float32,
            sigma: np.float32,
        ) -> MapValues:
            """
            Computes the gaussian coefficients for the update of the codebook. The coefficients are computed for all neurons in the codebook with respect to the BMU

            Parameters
            ----------
            map_distances : MapValues
                Map distances of all neurons in the codebook with respect to the BMU
            alpha : np.float32
                Current learning rate
            sigma : np.float32
                Current neighborhood size

            Returns
            -------
            MapValues
                Mexican coefficients for all neurons in the codebook
            """
            rows, columns = map_distances.shape
            coefficients = np.zeros((rows, columns), dtype=np.float32)
            f = 1 / (sigma * (40 / 9))
            for neuron_index in pndindex((rows, columns)):
                component_1 = 1 - (
                    2 * (np.pi**2) * (f**2) * (map_distances[neuron_index] ** 2)
                )
                component_2 = np.e ** (
                    -(np.pi**2) * (f**2) * (map_distances[neuron_index] ** 2)
                )
                coefficients[neuron_index] = alpha * component_1 * component_2
            return coefficients

        return mexican_coeff

    elif kernel == "cone":

        @jit(
            [
                "float32[:,::1](float32[:,::1], float32, float32)",
            ],
            fastmath=fastmath_flag,
            nopython=True,
            parallel=True,
        )
        def cone_coeff(
            map_distances: MapValues,
            alpha: np.float32,
            sigma: np.float32,
        ) -> MapValues:
            """
            Updates all neurons in the codebook according to a cone neighorhood distribution. Does not return an array, as the underlying array whos reference is passed is updated

            Parameters:
            -----------
            map_distances : MapValues
                Map distances of all neurons in the codebook with respect to the BMU
            alpha : np.float32
                Current learning rate
            sigma : np.float32
                Current neighborhood size

            Returns:
            --------
            MapValues
                Cone coefficients for all neurons in the codebook
            """
            rows, columns = map_distances.shape
            coefficients = np.zeros((rows, columns), dtype=np.float32)

            for neuron_idx in np.ndindex(rows, columns):
                component_1 = np.maximum(0, 1 - (map_distances[neuron_idx] / sigma))
                coefficients[neuron_idx] = alpha * component_1
            return coefficients

        return cone_coeff
    else:
        raise ValueError(f"Kernel {kernel} is not supported!")


def make_local_kernel_coefficient_function(
    kernel: str, fastmath_flag: bool = FASTMATH_FLAG
) -> CodebookCoefficientsFunction:
    """
    Creates a local kernel coefficient function for the specified kernel type.
    Other than the default coefficient function, this function uses a linear layout of the distances and a restricted radius for the gaussion-like kernel.

    Parameters
    ----------
    kernel : str
        The type of kernel to create the coefficient function for. Supported kernels are "gaussian", "mexican", and "cone".
    fastmath_flag : bool, optional
        Using reduced precision for faster calculation, by default FASTMATH_FLAG

    Returns
    -------
    CodebookCoefficientsFunction
        A function that computes the coefficients for the specified kernel type.

    Raises
    ------
    ValueError
        If the specified kernel type is not supported.
    """
    if kernel == "gaussian":

        @njit(
            [
                "float32[::1](float32[::1], float32, float32)",
            ],
            fastmath=fastmath_flag,
            parallel=True,
        )
        def gaussian_coeff(
            map_distances: Vector,
            alpha: np.float32,
            sigma: np.float32,
        ) -> MapValues:
            """
            Computes the gaussian coefficients for the update of the codebook.

            Parameters
            ----------
            map_distances : MapValues
                Map distances of neurons in the neighborhood with respect to the BMU
            alpha : np.float32
                Current learning rate
            sigma : np.float32
                Current neighborhood size

            Returns
            -------
            MapValues
                Gaussian coefficients for neurons in the neighborhood
            """
            coefficients = np.zeros(map_distances.shape, dtype=np.float32)
            for neuron_index in prange(map_distances.shape[0]):
                coefficients[neuron_index] = alpha * np.exp(
                    -((map_distances[neuron_index] ** 2) * (3 / sigma**2))
                )
            return coefficients

        return gaussian_coeff

    elif kernel == "mexican":

        @njit(
            [
                "float32[::1](float32[::1], float32, float32)",
            ],
            fastmath=fastmath_flag,
            parallel=True,
        )
        def mexican_coeff(
            map_distances: Vector,
            alpha: np.float32,
            sigma_sub: np.float32,
        ) -> MapValues:
            """
            Computes the mexican coefficients for the update of the codebook.

            Parameters
            ----------
            map_distances : MapValues
                Map distances of neurons in the neighborhood with respect to the BMU
            alpha : np.float32
                Current learning rate
            sigma : np.float32
                Current neighborhood size

            Returns
            -------
            MapValues
                Mexican coefficients for neurons in the neighborhood
            """
            coefficients = np.zeros(map_distances.shape, dtype=np.float32)
            f = 1 / (sigma_sub * (40 / 9))
            for neuron_index in prange(map_distances.shape[0]):
                component_1 = 1 - (
                    2 * (np.pi**2) * (f**2) * (map_distances[neuron_index] ** 2)
                )
                component_2 = np.e ** (
                    -(np.pi**2) * (f**2) * (map_distances[neuron_index] ** 2)
                )
                coefficients[neuron_index] = alpha * component_1 * component_2
            return coefficients

        return mexican_coeff

    elif kernel == "cone":

        @njit(
            [
                "float32[::1](float32[::1], float32, float32)",
            ],
            fastmath=fastmath_flag,
            parallel=True,
        )
        def cone_coeff(
            map_distances: Vector,
            alpha: np.float32,
            sigma_sub: np.float32,
        ) -> MapValues:
            """
            Computes the cone coefficients for the update of the codebook.

            Parameters:
            -----------
            map_distances : MapValues
                Map distances of neurons in the neighborhood with respect to the BMU
            alpha : np.float32
                Current learning rate
            sigma : np.float32
                Current neighborhood size

            Returns:
            --------
            MapValues
                Cone coefficients for neurons in the neighborhood
            """
            coefficients = np.zeros(map_distances.shape, dtype=np.float32)

            for neuron_index in prange(map_distances.shape[0]):
                component_1 = np.maximum(
                    0, 1 - (map_distances[neuron_index] / sigma_sub)
                )
                coefficients[neuron_index] = alpha * component_1
            return coefficients

        return cone_coeff
    else:
        raise ValueError(f"Kernel {kernel} is not supported!")


def make_update_function(fastmath_flag: bool = FASTMATH_FLAG) -> CodebookUpdateFunction:
    """
    Creates a function that updates the codebook inplace.

    Parameters
    ----------
    fastmath_flag : bool, optional
        Using reduced precision for faster calculation, by default FASTMATH_FLAG

    Returns
    -------
    CodebookUpdateFunction
        A function that updates the codebook inplace.
    """

    @njit(
        [
            "void(float32[:, :, ::1], float32[::1], float32[:, ::1], int32[::1])",
        ],
        fastmath=fastmath_flag,
        parallel=True,
    )
    def update_codebook(
        codebook: Codebook,
        element: Vector,
        coefficients: MapValues,
        bmu_pos: Position,
    ) -> None:
        """
        Updates all neurons in the codebook according to the value supplied by `coefficients`.
        Does not return an array, as the underlying array whos reference is passed is updated

        Parameters
        ----------
        codebook : Codebook
            View of the codebook to be updated
        element : Vector
            Data element the update is performed towards
        map_distances : MapValues
            Map distances of all neurons in the codebook with respect to the BMU
        alpha : np.float32
            Current learning rate
        sigma : np.float32
            Current neighborhood size
        """

        rows, columns, _ = codebook.shape
        for neuron_index in pndindex((rows, columns)):
            row, column = neuron_index
            offset_row = abs(bmu_pos[0] - row)
            offset_col = abs(bmu_pos[1] - column)
            coefficient = coefficients[offset_row, offset_col]
            for i in range(element.shape[0]):
                codebook[row, column, i] += coefficient * (
                    element[i] - codebook[row, column, i]
                )

    return update_codebook


def make_local_update_function(
    fastmath_flag: bool = FASTMATH_FLAG,
) -> LocalUpdateFunction:
    """
    Creates a function that updates the codebook inplace for local updates.

    Parameters
    ----------
    fastmath_flag : bool, optional
        Using reduced precision for faster calculation, by default FASTMATH_FLAG

    Returns
    -------
    LocalUpdateFunction
        A function that updates the codebook according to values supplied by `coefficients`.

    """

    @njit(
        ["void(float32[:, :, ::1], float32[::1], int32[:,::1], float32[::1])"],
        fastmath=fastmath_flag,
        parallel=True,
        error_model="numpy",
    )
    def update_codebook(
        codebook: Codebook,
        element: Vector,
        neighbors: Neighbors,
        coefficients: Vector,
    ) -> None:
        """
        Updates all neurons in the codebook according to a gaussian-like neighorhood distribution. Uses a
        restrict gaussian radius for faster update calculation. Does not return an array, as the underlying
        array whos reference was passed is updated

        Parameters
        ----------
        codebook : Codebook
            View of the codebook to be updated
        element : Vector
            Data element the update is performed towards
        neighbors : Neighbors
            2D array of shape (n, 2) containing the row and column indices
        relative_map_distances : Vector
            Map distances of the neighboring neurons with respect to the BMU
        alpha : np.float32
            Current learning rate
        sigma : np.float32
            Current neighborhood size
        """
        for neuron_index in prange(neighbors.shape[0]):
            row, column = neighbors[neuron_index]
            coefficient = coefficients[neuron_index]
            for i in range(element.shape[0]):
                codebook[row, column, i] += coefficient * (
                    element[i] - codebook[row, column, i]
                )

    return update_codebook
