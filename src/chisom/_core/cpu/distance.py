#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 00:33:33 2022

author: j.kaminski@uni-muenster.de for agkoch
"""

import warnings
from typing import Tuple

import numpy as np
from numba import njit, pndindex, prange  # type: ignore
from numba.core.errors import NumbaPerformanceWarning  # type: ignore
from numpy.typing import NDArray

from chisom._core.cpu.types import (
    BoundedDistanceFunction,
    BroadcastingDistanceFunction,
    MapDistanceFunction,
    Neighbors,
    PairwiseDistanceFunction,
    VectorDistanceFunction,
)
from chisom._core.types import Codebook, MapValues, Position, Vector

FASTMATH_FLAG = True

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def make_vector_distance_func(
    norm: str, fastmath_flag: bool = FASTMATH_FLAG
) -> VectorDistanceFunction:
    """
    Factory function to create a distance function for a given norm. Does not support broadcasting.

    Parameters
    ----------
    norm : str
        The norm to use for the distance calculation. Supported norms are:
        - "manhattan"
        - "euclidean"
        - "cosine"
        - "dot"
    fastmath_flag : bool, optional
        Wether to use fastmath with reduce accuracy, by default FASTMATH_FLAG

    Returns
    -------
    Callable
        Function that calculates the distance between the last dimension of a 2D or 3D array and a vector.

    Raises
    ------
    ValueError
        If the norm is not supported.
    """
    if norm == "manhattan":

        @njit(
            [
                "float32[:,::1](float32[:,:,::1], float32[::1])",
            ],
            fastmath=fastmath_flag,
            parallel=True,
            error_model="numpy",
        )
        def manhattan(X: Codebook, v: Vector) -> MapValues:
            """
            Returns manhattan distances of the last dimensions of a codebook 'X', to a
            given vector 'v'.

            Parameters
            ----------
            X : Codebook
                3D Array, with the last dimension of size m.
            v : Vector
                Vector of length m.

            Returns
            -------
            distances : MapValues
                Manhattan distances of codebook neurons to vector.

            """

            indeces = X.shape[:-1]
            distances = np.empty(indeces, dtype=X.dtype)

            for index in pndindex(indeces):
                # TODO: check for single-precision enforcing
                distances[index] = np.sum(np.abs(X[index] - v))

            return distances

        return manhattan

    elif norm == "euclidean":

        @njit(
            [
                "float32[:,::1](float32[:,:,::1], float32[::1])",
            ],
            fastmath=fastmath_flag,
            parallel=True,
            error_model="numpy",
        )
        def euclidean(X: Codebook, v: Vector) -> MapValues:
            """
            Returns euclidean distances of the last dimensions of a codebook 'X', to a
            given vector 'v'.

            Parameters
            ----------
            X : Codebook
                3D Array, with the last dimension of size m.
            v : Vector
                Vector of length m.

            Returns
            -------
            distances : MapValues
                Euclidean distances of codebook neurons to vector.
            """

            indeces = X.shape[:-1]
            distances = np.empty(indeces, dtype=np.float32)

            for index in pndindex(indeces):
                # TODO: check for single-precision enforcing
                distances[index] = np.sqrt(np.sum(np.power(X[index] - v, 2)))

            return distances

        return euclidean

    elif norm == "cosine":

        @njit(
            [
                "float32[:,::1](float32[:,:,::1], float32[::1])",
            ],
            fastmath=fastmath_flag,
            parallel=True,
            error_model="numpy",
        )
        def cosine(X: Codebook, v: Vector) -> MapValues:
            """
            Returns cosine distances of the last dimensions of a codebook 'X', to a
            given vector 'v'.

            Parameters
            ----------
            X : Codebook
                3D Array, with the last dimension of size m.
            v : Vector
                Vector of length m.

            Returns
            -------
            distances : MapValues
                Cosine distances of codebook neurons to vector.
            """
            indeces = X.shape[:-1]
            similarity = np.empty(indeces, np.float32)
            norm_v = np.sqrt(np.dot(v, v))

            for index in pndindex(indeces):
                similarity[index] = np.dot(X[index], v) / (
                    np.sqrt(np.dot(X[index], X[index])) * norm_v
                )

            return 1 - similarity

        return cosine

    elif norm == "dot":

        @njit(
            [
                "float32[:,::1](float32[:,:,::1], float32[::1])",
            ],
            fastmath=fastmath_flag,
            parallel=True,
            error_model="numpy",
        )
        def dot(X: Codebook, v: Vector) -> MapValues:
            """
            Returns dot distances of the last dimensions of a codebook 'X', to a
            given vector 'v'.

            Parameters
            ----------
            X : Codebook
                3D Array, with the last dimension of size m.
            v : Vector
                Vector of length m.

            Returns
            -------
            distances : MapValues
                Dot distances of codebook neurons to vector.
            """
            indeces = X.shape[:-1]
            similarity = np.empty(indeces, np.float32)

            for index in pndindex(indeces):
                similarity[index] = np.dot(X[index], v)

            return similarity

        return dot

    else:
        raise ValueError("Distance norm unknown")


def make_universal_distance_func(
    norm: str, fastmath_flag: bool = FASTMATH_FLAG
) -> BroadcastingDistanceFunction:
    """
    Factory function to create a universal distance function that broadcast to most matchups of shapes based on the specified norm.

    Parameters
    ----------
    norm : str
        The norm to use for the distance calculation. Supported norms are:
        - "manhattan"
        - "euclidean"
        - "cosine"
        - "dot"
    fastmath_flag : bool, optional
        Wether to use fastmath with reduce accuracy, by default FASTMATH_FLAG

    Returns
    -------
    BroadcastingDistanceFunction
        Function that calculates the distance between the last dimension of two arrays.

    Raises
    ------
    ValueError
        If the norm is not supported.
    """

    if norm == "manhattan":

        @njit(
            [
                "float32(float32[::1], float32[::1])",
                "float32[::1](float32[:,::1], float32[::1])",
                "float32[:,::1](float32[:,:,::1], float32[::1])",
                "float32[::1](float32[:,::1], float32[:,::1])",
                "float32[:,::1](float32[:,:,::1], float32[:,:,::1])",
            ],
            fastmath=fastmath_flag,
            parallel=True,
            error_model="numpy",
        )
        def manhattan(
            X_1: NDArray[np.float32], X_2: NDArray[np.float32]
        ) -> NDArray[np.float32]:
            """
            Returns manhattan distances of the last dimensions two given Matrices'X_1' and 'X_2'.
            If the the shapes do not match,X_2 should be the smaller one and will be broadcasted to the shape of X_1.

            Parameters
            ----------
            X_1 : NDArray[np.float32]
                1D, 2D or 3D Array, with the last dimension of size m.
            X_2 : NDArray[np.float32]
                1D or 3D Array, with the last dimension of size m.

            Returns
            -------
            distance : NDArray[np.float32]
                Array of manhattan distances as an array with one less dimension than X_1

            """
            assert X_1.shape[-1] == X_2.shape[-1], (
                "Last dimension of X_1 and X_2 must match"
            )
            # Add a new axis to X_1 and X_2, as pndindex expects currently does not support () for 1D arrays as a shape
            X_1 = X_1[np.newaxis, :]
            X_2 = X_2[np.newaxis, :]

            X_2 = np.broadcast_to(X_2, X_1.shape)
            indeces = X_1.shape[:-1]
            distances = np.empty(indeces, dtype=X_1.dtype)

            for index in pndindex(indeces):
                # TODO: check for single-precision enforcing
                distances[index] = np.sum(np.abs(X_1[index] - X_2[index]))

            return distances[0]  # Remove the new axis added at the beginning

        return manhattan

    elif norm == "euclidean":

        @njit(
            [
                "float32(float32[::1], float32[::1])",
                "float32[::1](float32[:,::1], float32[::1])",
                "float32[:,::1](float32[:,:,::1], float32[::1])",
                "float32[::1](float32[:,::1], float32[:,::1])",
                "float32[:,::1](float32[:,:,::1], float32[:,:,::1])",
            ],
            fastmath=fastmath_flag,
            parallel=True,
            error_model="numpy",
        )
        def euclidean(
            X_1: NDArray[np.float32], X_2: NDArray[np.float32]
        ) -> NDArray[np.float32]:
            """
            Returns euclidean distances of the last dimensions two given Matrices'X_1' and 'X_2'.
            If the the shapes do not match,X_2 should be the smaller one and will be broadcasted to the shape of X_1.

            Parameters
            ----------
            X_1 : NDArray[np.float32]
                1D, 2D or 3D Array, with the last dimension of size m.
            X_2 : NDArray[np.float32]
                1D or 3D Array, with the last dimension of size m.

            Returns
            -------
            distance : NDArray[np.float32]
                Array of euclidean distances as an array with one less dimension than X_1

            """
            assert X_1.shape[-1] == X_2.shape[-1], (
                "Last dimension of X_1 and X_2 must match"
            )
            # Add a new axis to X_1 and X_2, as pndindex expects currently does not support () for 1D arrays as a shape
            X_1 = X_1[np.newaxis, :]
            X_2 = X_2[np.newaxis, :]

            X_2 = np.broadcast_to(X_2, X_1.shape)
            indeces = X_1.shape[:-1]
            distances = np.empty(indeces, dtype=X_1.dtype)

            for index in pndindex(indeces):
                # TODO: check for single-precision enforcing
                distances[index] = np.sqrt(np.sum(np.power(X_1[index] - X_2[index], 2)))

            return distances[0]  # Remove the new axis added at the beginning

        return euclidean

    elif norm == "cosine":

        @njit(
            [
                "float32(float32[::1], float32[::1])",
                "float32[::1](float32[:,::1], float32[::1])",
                "float32[:,::1](float32[:,:,::1], float32[::1])",
                "float32[::1](float32[:,::1], float32[:,::1])",
                "float32[:,::1](float32[:,:,::1], float32[:,:,::1])",
            ],
            fastmath=fastmath_flag,
            parallel=True,
            # error_model="numpy",
        )
        def cosine(
            X_1: NDArray[np.float32], X_2: NDArray[np.float32]
        ) -> NDArray[np.float32]:
            """
            Returns cosine distances of the last dimensions two given Matrices'X_1' and 'X_2'.
            If the the shapes do not match,X_2 should be the smaller one and will be broadcasted to the shape of X_1.

            Parameters
            ----------
            X_1 : NDArray[np.float32]
                1D, 2D or 3D Array, with the last dimension of size m.
            X_2 : NDArray[np.float32]
                1D or 3D Array, with the last dimension of size m.

            Returns
            -------
            distance : NDArray[np.float32]
                Array of manhattan cosine as an array with one less dimension than X_1

            """
            assert X_1.shape[-1] == X_2.shape[-1], (
                "Last dimension of X_1 and X_2 must match"
            )
            # Add a new axis to X_1 and X_2, as pndindex expects currently does not support () for 1D arrays as a shape
            X_1 = X_1[np.newaxis, :]
            X_2 = X_2[np.newaxis, :]

            # Calculate the scaled dot product for the matrix X_2
            # This is done as an extra step, as when the shape of X_2 is smaller than X_1,
            # broadcasting leads to less compoutation
            dot_out_shape = X_2.shape[:-1]
            dot_out = np.empty(dot_out_shape, np.float32)
            for index in pndindex(dot_out_shape):
                dot_out[index] = np.sum(X_2[index] * X_2[index])
            scale_v = np.sqrt(dot_out)

            X_2 = np.broadcast_to(X_2, X_1.shape)
            out_shape = X_1.shape[:-1]
            scale_v = np.broadcast_to(scale_v, out_shape)

            similarity = np.empty(out_shape, np.float32)

            for index in pndindex(out_shape):
                similarity[index] = np.dot(X_1[index], X_2[index]) / (
                    np.sqrt(np.dot(X_1[index], X_1[index])) * scale_v[index]
                )

            return 1 - similarity[0]

        return cosine

    elif norm == "dot":

        @njit(
            [
                "float32(float32[::1], float32[::1])",
                "float32[::1](float32[:,::1], float32[::1])",
                "float32[:,::1](float32[:,:,::1], float32[::1])",
                "float32[::1](float32[:,::1], float32[:,::1])",
                "float32[:,::1](float32[:,:,::1], float32[:,:,::1])",
            ],
            fastmath=fastmath_flag,
            parallel=True,
            error_model="numpy",
        )
        def dot(
            X_1: NDArray[np.float32], X_2: NDArray[np.float32]
        ) -> NDArray[np.float32]:
            """
            Returns dot product of the last dimensions two given Matrices'X_1' and 'X_2'.
            If the the shapes do not match,X_2 should be the smaller one and will be broadcasted to the shape of X_1.

            Parameters
            ----------
            X_1 : NDArray[np.float32]
                1D, 2D or 3D Array, with the last dimension of size m.
            X_2 : NDArray[np.float32]
                1D or 3D Array, with the last dimension of size m.

            Returns
            -------
            distance : NDArray[np.float32]
                Array of dot product as an array with one less dimension than X_1

            """
            # Add a new axis to X_1 and X_2, as pndindex expects currently does not support () for 1D arrays as a shape
            X_1 = X_1[np.newaxis, :]
            X_2 = X_2[np.newaxis, :]

            X_2 = np.broadcast_to(X_2, X_1.shape)
            indeces = X_1.shape[:-1]
            similarity = np.empty(indeces, dtype=X_1.dtype)

            for index in pndindex(indeces):
                similarity[index] = np.sum(X_1[index] * X_2[index])

            return similarity[0]  # Remove the new axis added at the beginning

        return dot
    else:
        raise ValueError("Distance norm unknown")


def make_map_distance_function(
    norm: str, fastmath_flag: bool = FASTMATH_FLAG
) -> MapDistanceFunction:
    """
    Factory function to create a map distance function that calculates the distance of all points on a grid to a reference point.
    Only uses information about grid size and position of the reference point to find distance.

    Parameters
    ----------
    norm : str
        The norm to use for the distance calculation. Supported norms are:
        - "manhattan"
        - "euclidean"
        - "manhattan_toroid"
        - "euclidean_toroid"
    fastmath_flag : bool, optional
        Wether to use fastmath with reduce accuracy, by default FASTMATH_FLAG

    Returns
    -------
    MapDistanceFunction
        Function that calculates the distance of all points on a grid to a reference point.
        Only uses information about grid size and position of the reference point to find distance.

    Raises
    ------
    ValueError
        If the norm is not supported.
    """
    if norm == "manhattan":

        @njit(
            ["float32[:,::1](int32[::1], int32[::1])"],
            fastmath=fastmath_flag,
            parallel=True,
        )
        def manhattan_boundary(reference: Position, grid_size: Position) -> MapValues:
            """
            Returns the manhattan distances to a point 'reference' for a all points on a grid, given the size of the grid 'grid_size'.

            Parameters
            ----------
            reference : Position
                Point on grid
            grid_size : Position
                Size of the grid

            Returns
            -------
            MapValues
                Manhattan distance for each point on the grid with regards to the reference point
            """
            tuple_grid_size = (grid_size[0], grid_size[1])
            distances = np.empty(tuple_grid_size, dtype=np.float32)

            for index in pndindex(tuple_grid_size):
                delta = np.absolute(reference - np.array(index, dtype=np.float32))
                distances[index] = np.sum(delta)

            return distances

        return manhattan_boundary

    elif norm == "euclidean":

        @njit(
            ["float32[:,::1](int32[::1], int32[::1])"],
            fastmath=fastmath_flag,
            parallel=True,
        )
        def euclidean_boundary(position: Position, grid_size: Position) -> MapValues:
            """
            Returns the euclidean distances to a point 'position' for all points on a grid, given the size of the grid 'grid_size'.

            Parameters
            ----------
            position : Position
                Point on grid
            grid_size : Position
                Sizes of the dimensions of the grid

            Returns
            -------
            MapValues
                Euclidean distance for each point on the grid with regards to the reference point
            """
            tuple_grid_size = (grid_size[0], grid_size[1])
            distances = np.empty(tuple_grid_size, dtype=np.float32)

            for index in pndindex(tuple_grid_size):
                delta = position - np.array(index, dtype=np.float32)
                distances[index] = np.sqrt(np.sum(np.power(delta, 2)))

            return distances

        return euclidean_boundary

    elif norm == "manhattan_toroid":

        @njit(
            ["float32[:,::1](int32[::1], int32[::1])"],
            fastmath=fastmath_flag,
            parallel=True,
        )
        def manhattan_toroid(reference: Position, grid_size: Position) -> Position:
            """
            Returns the manhattan distances to a point 'reference' for a all points on a toroidal grid, given the size of the grid 'grid_size'.

            Parameters
            ----------
            reference : Position
                Point on grid
            grid_size : Position
                Sizes of the dimensions of the grid

            Returns
            -------
            Position
                Manhattan distance for each point on the toroidal grid with regards to the reference point
            """
            tuple_grid_size = (grid_size[0], grid_size[1])
            distances = np.empty(tuple_grid_size, dtype=np.float32)

            for index in pndindex(tuple_grid_size):
                # TODO: check for single-precision enforcing
                delta = np.absolute(reference - np.array(index, dtype=np.float32))
                distances[index] = np.sum(np.minimum(delta, grid_size - delta))

            return distances

        return manhattan_toroid

    elif norm == "euclidean_toroid":

        @njit(
            ["float32[:,::1](int32[::1], int32[::1])"],
            fastmath=fastmath_flag,
            parallel=True,
        )
        def euclidean_toroid(position: Position, grid_size: Position) -> MapValues:
            """
            Returns the euclidean distances to a point 'position' for all points on a toroidal grid, given the size of the grid 'grid_size'.

            Parameters
            ----------
            position : Position
                Point on grid
            grid_size : Position
                Sizes of the dimensions of the grid

            Returns
            -------
            MapValues
                Euclidean distance for each point on the toroidal grid with regards to the reference point
            """
            tuple_grid_size = (grid_size[0], grid_size[1])
            distances = np.empty(tuple_grid_size, dtype=np.float32)

            for index in pndindex(tuple_grid_size):
                delta = np.absolute(position - np.array(index, dtype=np.float32))
                distances[index] = np.sqrt(
                    np.sum(np.power(np.minimum(delta, grid_size - delta), 2))
                )

            return distances

        return euclidean_toroid

    else:
        raise ValueError("Map distance norm unknown")


def make_bounded_distance_func(
    norm: str, fastmath_flag: bool = FASTMATH_FLAG
) -> BoundedDistanceFunction:
    """
    Factory function to create function that returns the relative indices and distances of all points on a grid,
    that would be within the specified norm distance of radius 'sigma' to a center point.

    Parameters
    ----------
    norm : str
        The norm to use for the distance calculation. Supported norms are:
        - "manhattan"
        - "manhattan_toroid"
        - "euclidean"
        - "euclidean_toroid"
    fastmath_flag : bool, optional
        Wether to use fastmath with reduce accuracy, by default FASTMATH_FLAG

    Returns
    -------
    BoundedDistanceFunction
        Function that calculates the relative indices and distances of all points on a grid that would be within
        the specified norm distance of radius 'sigma' to a center point.

    Raises
    ------
    ValueError
        If the norm is not supported.
    """
    if norm == "manhattan" or norm == "manhattan_toroid":

        @njit(
            "Tuple((int32[:,::1], float32[::1]))(float32)",
            parallel=True,
            fastmath=fastmath_flag,
        )
        def relative_manhattan_neigborhood(
            sigma: np.float32,
        ) -> Tuple[Neighbors, Vector]:
            """
            Create relative indeces and distances of all points on a grid that would be within the manhattan distance of radius 'sigma' to a center point.

            Parameters
            ----------
            sigma : np.float32
                Neighborhood radius

            Returns
            -------
            Tuple[NDArray[np.int32], NDArray[np.int32]]
                Relative Indices and distances
            """
            # Using fancy broadcasting to make a grid and calculate distances
            sigma = np.floor(sigma)
            delta = np.arange(-sigma, sigma + 1, dtype=np.int32)
            delta_rows = delta[:, np.newaxis]
            delta_columns = delta[np.newaxis, :]
            delta_rows, delta_columns = np.broadcast_arrays(delta_rows, delta_columns)
            distances = np.asarray(
                np.absolute(delta_rows) + np.absolute(delta_columns), dtype=np.float32
            )

            # usa a mask to only retreive point where distance criteria are met
            mask = distances <= sigma
            mask = mask.flatten()
            relative_neighbors = np.column_stack(
                (delta_rows.flatten()[mask], delta_columns.flatten()[mask])
            )

            return relative_neighbors, distances.flatten()[mask]

        return relative_manhattan_neigborhood

    elif norm == "euclidean" or norm == "euclidean_toroid":

        @njit(
            "Tuple((int32[:,::1],float32[::1]))(float32)",
            parallel=True,
            fastmath=fastmath_flag,
        )
        def relative_euclidean_neigborhood(
            sigma: np.float32,
        ) -> Tuple[Neighbors, Vector]:
            """
            Create relative indeces and distances of all points on a grid that would be within the euclidean distance of radius 'sigma' to a center point.

            Parameters
            ----------
            sigma : np.float32
                Neighborhood radius

            Returns
            -------
            Tuple[Neighbors, Vector]
                Relative Indices and distances
            """

            # Using fancy broadcasting to make a grid and calculate distances
            sigma_ceil = np.ceil(sigma)
            delta = np.arange(-sigma_ceil, sigma_ceil + 1, dtype=np.int32)
            delta_rows = delta[:, np.newaxis]
            delta_columns = delta[np.newaxis, :]
            delta_rows, delta_columns = np.broadcast_arrays(delta_rows, delta_columns)

            # Using sqrt here as later needed in update calculation anyways
            distances = np.asarray(
                np.sqrt(delta_rows**2 + delta_columns**2), dtype=np.float32
            )

            # usa a mask to only retreive point where distance criteria are met
            mask = distances <= sigma
            mask = mask.flatten()
            relative_neighbors = np.column_stack(
                (delta_rows.flatten()[mask], delta_columns.flatten()[mask])
            )

            return relative_neighbors, distances.flatten()[mask]

        return relative_euclidean_neigborhood
    else:
        raise ValueError("Map distance norm unknown")


def make_pairwise_distance_func(
    norm: str, fastmath_flag: bool = FASTMATH_FLAG
) -> PairwiseDistanceFunction:
    if norm == "manhattan":

        @njit(
            [
                "float32[:,::1](float32[:,::1], int32[::1])",
                "float32[:,::1](int32[:,::1], int32[::1])",
                "float32[:,::1](int16[:,::1], int32[::1])",
            ],
            fastmath=fastmath_flag,
            parallel=True,
        )
        def manhattan(
            M: NDArray[np.int32 | np.float32], grid_size: NDArray[np.int32]
        ) -> NDArray[np.float32]:
            """
            Returns manhattan distances of the last dimensions of a 2D or 3D array 'X', to a
            given vector 'v'.

            Parameters
            ----------
            X : NDArray[np.int32  |  np.float32]
                2D/3D Array, with the last dimension of size m.
            v : NDArray[np.int32  |  np.float32]
                Vector of length m.

            Returns
            -------
            NDArray[np.float32]
                Euclidean distances with one less dimension than 'X'.
            """

            m = M.shape[0]

            output_shape = (m, m)
            distances = np.zeros(output_shape, dtype=np.float32)

            for i in prange(m):
                for j in prange(i + 1, m):
                    res = np.sum(np.abs(M[i] - M[j]))
                    distances[i, j] = res

            distances += distances.T

            return distances

        return manhattan

    elif norm == "euclidean":

        @njit(
            [
                "float32[:,::1](float32[:,::1], int32[::1])",
                "float32[:,::1](int32[:,::1], int32[::1])",
                "float32[:,::1](int16[:,::1], int32[::1])",
            ],
            fastmath=fastmath_flag,
            parallel=True,
        )
        def euclidean(
            M: NDArray[np.int32 | np.float32], grid_size: NDArray[np.int32]
        ) -> NDArray[np.float32]:
            """
            Returns euclidean distances of the last dimensions of a 2D or 3D array 'X', to a
            given vector 'v'.

            Parameters
            ----------
            X : NDArray[np.int32  |  np.float32]
                2D/3D Array, with the last dimension of size m.
            v : NDArray[np.int32  |  np.float32]
                Vector of length m.

            Returns
            -------
            NDArray[np.float32]
                Euclidean distances with one less dimension than 'X'.
            """

            m = M.shape[0]

            output_shape = (m, m)
            distances = np.zeros(output_shape, dtype=np.float32)

            for i in prange(m):
                for j in prange(i + 1, m):
                    res = np.sqrt(np.sum(np.power(M[i] - M[j], 2)))
                    distances[i, j] = res

            distances += distances.T

            return distances

        return euclidean

    elif norm == "manhattan_toroid":

        @njit(
            [
                "float32[:,::1](float32[:,::1], int32[::1])",
                "float32[:,::1](int32[:,::1], int32[::1])",
                "float32[:,::1](int16[:,::1], int32[::1])",
            ],
            fastmath=fastmath_flag,
            parallel=True,
        )
        def manhattan_toroid(
            M: NDArray[np.int32 | np.float32], grid_size: NDArray[np.int32]
        ) -> NDArray[np.float32]:
            m = M.shape[0]

            output_shape = (m, m)
            distances = np.zeros(output_shape, dtype=np.float32)

            for i in prange(m):
                for j in prange(i + 1, m):
                    delta = np.absolute(M[i] - M[j])
                    res = np.sum(np.minimum(delta, grid_size - delta))
                    distances[i, j] = res

            distances += distances.T

            return distances

        return manhattan_toroid

    elif norm == "euclidean_toroid":

        @njit(
            [
                "float32[:,::1](float32[:,::1], int32[::1])",
                "float32[:,::1](int32[:,::1], int32[::1])",
                "float32[:,::1](int16[:,::1], int32[::1])",
            ],
            fastmath=fastmath_flag,
            parallel=True,
        )
        def euclidean_toroid(
            M: NDArray[np.int32 | np.float32], grid_size: NDArray[np.int32]
        ) -> NDArray[np.float32]:
            m = M.shape[0]

            output_shape = (m, m)
            distances = np.zeros(output_shape, dtype=np.float32)

            for i in prange(m):
                for j in prange(i + 1, m):
                    delta = np.absolute(M[i] - M[j])
                    res = np.sqrt(
                        np.sum(np.power(np.minimum(delta, grid_size - delta), 2))
                    )
                    distances[i, j] = res

            distances += distances.T

            return distances

        return euclidean_toroid
    else:
        raise ValueError("Pairwise distance norm unknown")
