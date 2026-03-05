import math

import numpy as np
from numba import cuda, float32
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from chisom._core.types import MapValues, Position

"""
CUDA Argmin implementation heavily inspired by GBJim's argmax_example.cu
https://gist.github.com/GBJim/7ea9c16e9bea5b4dc913246e2e2dbc84
"""


def cuda_argmin_2d(
    distance_matrix: MapValues, partial_argmin: MapValues, array_size: Position, stream
) -> DeviceNDArray:
    """
    Compute the argmin of a 2D array on the GPU using a parallel reduction approach.

    Parameters
    ----------
    distance_matrix : MapValues
        The input distance matrix to find the minimum value in
    partial_argmin : MapValues
        Device array to store intermediate and final results
    array_size : Position
        The size of the input array
    stream : cuda.Stream
        CUDA stream to use for execution

    Returns
    -------
    Position
        The position (row, column) of the minimum value in the input array
    """
    threads_per_block = (32, 32)
    array_dimensions_shape = distance_matrix.shape
    new_array_dimensions = (  # Derived from block size, since each block is reduced to a single value
        (array_dimensions_shape[0] // threads_per_block[0]) + 1,
        (array_dimensions_shape[1] // threads_per_block[1]) + 1,
    )

    cuda_argmin_2d_partial_build_index[new_array_dimensions, threads_per_block, stream](
        distance_matrix, partial_argmin, array_dimensions_shape
    )

    while new_array_dimensions[0] > 1 or new_array_dimensions[1] > 1:
        array_dimensions = np.array(new_array_dimensions, dtype=np.int32)
        new_array_dimensions = (
            int((array_dimensions[0] // threads_per_block[0]) + 1),
            int((array_dimensions[1] // threads_per_block[1]) + 1),
        )
        cuda_argmin_2d_partial_existing_index[
            new_array_dimensions, threads_per_block, stream
        ](distance_matrix, partial_argmin, array_dimensions)

    return partial_argmin[0, 0]


@cuda.jit
def cuda_argmin_2d_partial_build_index(
    distance_matrix: MapValues, partial_argmin: MapValues, array_dimensions: Position
):
    """
    Used during the first iteration of argmin to build indices for the whole distance matrix.

    Parameters
    ----------
    distance_matrix : MapValues
        DeviceArray containing the distance values and used to return minimum value of thread block
    partial_argmin : MapValues
        DeviceArray to return the index of the minimum element of a thread block
    array_dimensions : Position
        Dimensions of the input array to determine bounds of the operation
    """

    row_grid, column_grid = cuda.grid(2)
    row_grid_size, column_grid_size = cuda.gridsize(2)
    thread_min_value = math.inf
    thread_min_row = row_grid
    thread_min_column = column_grid

    # Grid-stride loop to find minimum value and its position
    for column_idx in range(column_grid, array_dimensions[1], column_grid_size):
        for row_idx in range(row_grid, array_dimensions[0], row_grid_size):
            if thread_min_value > distance_matrix[row_idx, column_idx]:
                thread_min_value = distance_matrix[row_idx, column_idx]
                thread_min_row = row_idx
                thread_min_column = column_idx

    # Shared memory for parallel reduction
    shared_min_values = cuda.shared.array(1024, dtype=float32)
    shared_min_rows = cuda.shared.array(1024, dtype=float32)
    shared_min_columns = cuda.shared.array(1024, dtype=float32)

    # Populate shared memory
    thread_idx = cuda.threadIdx.x + (cuda.blockDim.x * cuda.threadIdx.y)
    shared_min_values[thread_idx] = thread_min_value
    shared_min_rows[thread_idx] = thread_min_row
    shared_min_columns[thread_idx] = thread_min_column

    cuda.syncthreads()

    # Parallel reduction to find block-wide minimum
    stride = (cuda.blockDim.x * cuda.blockDim.y) // 2
    while stride > 32:
        if thread_idx < stride:
            if shared_min_values[thread_idx] > shared_min_values[thread_idx + stride]:
                shared_min_values[thread_idx] = shared_min_values[thread_idx + stride]
                shared_min_rows[thread_idx] = shared_min_rows[thread_idx + stride]
                shared_min_columns[thread_idx] = shared_min_columns[thread_idx + stride]
        cuda.syncthreads()
        stride = stride // 2

    # Warp-level reduction (unrolled for efficiency)
    if thread_idx < 32:
        if shared_min_values[thread_idx] > shared_min_values[thread_idx + 32]:
            shared_min_values[thread_idx] = shared_min_values[thread_idx + 32]
            shared_min_rows[thread_idx] = shared_min_rows[thread_idx + 32]
            shared_min_columns[thread_idx] = shared_min_columns[thread_idx + 32]
        if shared_min_values[thread_idx] > shared_min_values[thread_idx + 16]:
            shared_min_values[thread_idx] = shared_min_values[thread_idx + 16]
            shared_min_rows[thread_idx] = shared_min_rows[thread_idx + 16]
            shared_min_columns[thread_idx] = shared_min_columns[thread_idx + 16]
        if shared_min_values[thread_idx] > shared_min_values[thread_idx + 8]:
            shared_min_values[thread_idx] = shared_min_values[thread_idx + 8]
            shared_min_rows[thread_idx] = shared_min_rows[thread_idx + 8]
            shared_min_columns[thread_idx] = shared_min_columns[thread_idx + 8]
        if shared_min_values[thread_idx] > shared_min_values[thread_idx + 4]:
            shared_min_values[thread_idx] = shared_min_values[thread_idx + 4]
            shared_min_rows[thread_idx] = shared_min_rows[thread_idx + 4]
            shared_min_columns[thread_idx] = shared_min_columns[thread_idx + 4]
        if shared_min_values[thread_idx] > shared_min_values[thread_idx + 2]:
            shared_min_values[thread_idx] = shared_min_values[thread_idx + 2]
            shared_min_rows[thread_idx] = shared_min_rows[thread_idx + 2]
            shared_min_columns[thread_idx] = shared_min_columns[thread_idx + 2]
        if shared_min_values[thread_idx] > shared_min_values[thread_idx + 1]:
            shared_min_values[thread_idx] = shared_min_values[thread_idx + 1]
            shared_min_rows[thread_idx] = shared_min_rows[thread_idx + 1]
            shared_min_columns[thread_idx] = shared_min_columns[thread_idx + 1]

    # Store block result
    if thread_idx == 0:
        partial_argmin[cuda.blockIdx.x, cuda.blockIdx.y, 0] = shared_min_rows[0]
        partial_argmin[cuda.blockIdx.x, cuda.blockIdx.y, 1] = shared_min_columns[0]
        distance_matrix[cuda.blockIdx.x, cuda.blockIdx.y] = shared_min_values[0]


@cuda.jit
def cuda_argmin_2d_partial_existing_index(
    distance_matrix: MapValues, partial_argmin: MapValues, array_dimensions: Position
):
    """
    Used during subsequent iterations of argmin to use existing indices from previous iterations.

    Parameters
    ----------
    distance_matrix : MapValues
        DeviceArray containing the distance values and used to return minimum value of thread block
    partial_argmin : MapValues
        DeviceArray to return the index of the minimum element of a thread block
    array_dimensions : Position
        Dimensions of the input array to determine bounds of the operation
    """

    row_grid, column_grid = cuda.grid(2)
    row_grid_size, column_grid_size = cuda.gridsize(2)
    thread_min_value = math.inf
    thread_min_row = row_grid
    thread_min_column = column_grid

    # Grid-stride loop to find minimum value and its corresponding indices
    for column_idx in range(column_grid, array_dimensions[1], column_grid_size):
        for row_idx in range(row_grid, array_dimensions[0], row_grid_size):
            if thread_min_value > distance_matrix[row_idx, column_idx]:
                thread_min_value = distance_matrix[row_idx, column_idx]
                thread_min_row = partial_argmin[row_idx, column_idx, 0]
                thread_min_column = partial_argmin[row_idx, column_idx, 1]

    # Shared memory for parallel reduction
    shared_min_values = cuda.shared.array(1024, dtype=float32)
    shared_min_rows = cuda.shared.array(1024, dtype=float32)
    shared_min_columns = cuda.shared.array(1024, dtype=float32)

    # Populate shared memory
    thread_idx = cuda.threadIdx.x + (cuda.blockDim.x * cuda.threadIdx.y)
    shared_min_values[thread_idx] = thread_min_value
    shared_min_rows[thread_idx] = thread_min_row
    shared_min_columns[thread_idx] = thread_min_column

    cuda.syncthreads()

    # Parallel reduction to find block-wide minimum
    stride = (cuda.blockDim.x * cuda.blockDim.y) // 2
    while stride > 32:
        if thread_idx < stride:
            if shared_min_values[thread_idx] > shared_min_values[thread_idx + stride]:
                shared_min_values[thread_idx] = shared_min_values[thread_idx + stride]
                shared_min_rows[thread_idx] = shared_min_rows[thread_idx + stride]
                shared_min_columns[thread_idx] = shared_min_columns[thread_idx + stride]
        cuda.syncthreads()
        stride = stride // 2

    # Warp-level reduction (unrolled for efficiency)
    if thread_idx < 32:
        if shared_min_values[thread_idx] > shared_min_values[thread_idx + 32]:
            shared_min_values[thread_idx] = shared_min_values[thread_idx + 32]
            shared_min_rows[thread_idx] = shared_min_rows[thread_idx + 32]
            shared_min_columns[thread_idx] = shared_min_columns[thread_idx + 32]
        if shared_min_values[thread_idx] > shared_min_values[thread_idx + 16]:
            shared_min_values[thread_idx] = shared_min_values[thread_idx + 16]
            shared_min_rows[thread_idx] = shared_min_rows[thread_idx + 16]
            shared_min_columns[thread_idx] = shared_min_columns[thread_idx + 16]
        if shared_min_values[thread_idx] > shared_min_values[thread_idx + 8]:
            shared_min_values[thread_idx] = shared_min_values[thread_idx + 8]
            shared_min_rows[thread_idx] = shared_min_rows[thread_idx + 8]
            shared_min_columns[thread_idx] = shared_min_columns[thread_idx + 8]
        if shared_min_values[thread_idx] > shared_min_values[thread_idx + 4]:
            shared_min_values[thread_idx] = shared_min_values[thread_idx + 4]
            shared_min_rows[thread_idx] = shared_min_rows[thread_idx + 4]
            shared_min_columns[thread_idx] = shared_min_columns[thread_idx + 4]
        if shared_min_values[thread_idx] > shared_min_values[thread_idx + 2]:
            shared_min_values[thread_idx] = shared_min_values[thread_idx + 2]
            shared_min_rows[thread_idx] = shared_min_rows[thread_idx + 2]
            shared_min_columns[thread_idx] = shared_min_columns[thread_idx + 2]
        if shared_min_values[thread_idx] > shared_min_values[thread_idx + 1]:
            shared_min_values[thread_idx] = shared_min_values[thread_idx + 1]
            shared_min_rows[thread_idx] = shared_min_rows[thread_idx + 1]
            shared_min_columns[thread_idx] = shared_min_columns[thread_idx + 1]

    # Store block result
    if thread_idx == 0:
        partial_argmin[cuda.blockIdx.x, cuda.blockIdx.y, 0] = shared_min_rows[0]
        partial_argmin[cuda.blockIdx.x, cuda.blockIdx.y, 1] = shared_min_columns[0]
        distance_matrix[cuda.blockIdx.x, cuda.blockIdx.y] = shared_min_values[0]
