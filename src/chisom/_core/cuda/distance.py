import math
from typing import Optional

from numba import cuda, types
from numba.cuda.dispatcher import CUDADispatcher

from chisom._core.cuda.types import GPUDeviceDistance, VectorDistanceFunction
from chisom._core.types import (
    Codebook,
    MapValues,
    Vector,
)


def _manhattan(codebook: Codebook, input_vector: Vector, distance_matrix: MapValues):
    row_grid_position, column_grid_position = cuda.grid(2)
    row_grid_size, column_grid_size = cuda.gridsize(2)
    num_rows, num_columns, num_features = codebook.shape

    shared_vector = cuda.shared.array(2048, dtype=types.float32)
    for feature_idx in range(
        (cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x),
        num_features,
        (cuda.blockDim.x * cuda.blockDim.y),
    ):
        shared_vector[feature_idx] = input_vector[feature_idx]
    cuda.syncthreads()

    for row_idx in range(row_grid_position, num_rows, row_grid_size):
        for column_idx in range(column_grid_position, num_columns, column_grid_size):
            distance_sum = types.float32(0.0)
            for feature_idx in range(num_features):
                distance_sum += math.fabs(
                    codebook[row_idx, column_idx, feature_idx]
                    - shared_vector[feature_idx]
                )
            distance_matrix[row_idx, column_idx] = distance_sum


def _euclidean(codebook: Codebook, input_vector: Vector, distance_matrix: MapValues):
    row_grid_position, column_grid_position = cuda.grid(2)
    row_grid_size, column_grid_size = cuda.gridsize(2)
    num_rows, num_columns, num_features = codebook.shape

    shared_vector = cuda.shared.array(2048, dtype=types.float32)
    for feature_idx in range(
        (cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x),
        num_features,
        (cuda.blockDim.x * cuda.blockDim.y),
    ):
        shared_vector[feature_idx] = input_vector[feature_idx]
    cuda.syncthreads()

    for row_idx in range(row_grid_position, num_rows, row_grid_size):
        for column_idx in range(column_grid_position, num_columns, column_grid_size):
            squared_distance_sum = types.float32(0.0)
            for feature_idx in range(num_features):
                difference = (
                    codebook[row_idx, column_idx, feature_idx]
                    - shared_vector[feature_idx]
                )
                squared_distance_sum += difference * difference
            distance_matrix[row_idx, column_idx] = math.sqrt(squared_distance_sum)


def _cosine(codebook: Codebook, input_vector: Vector, distance_matrix: MapValues):
    row_grid_position, column_grid_position = cuda.grid(2)
    row_grid_size, column_grid_size = cuda.gridsize(2)
    num_rows, num_columns, num_features = codebook.shape

    shared_vector = cuda.shared.array(2048, dtype=types.float32)
    for feature_idx in range(
        (cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x),
        num_features,
        (cuda.blockDim.x * cuda.blockDim.y),
    ):
        shared_vector[feature_idx] = input_vector[feature_idx]
    cuda.syncthreads()

    shared_vector_norm = cuda.shared.array(1, dtype=types.float32)
    if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
        shared_vector_norm[0] = types.float32(0.0)
    cuda.syncthreads()

    partial_norm_sum = types.float32(0.0)
    for feature_idx in range(
        (cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x),
        num_features,
        (cuda.blockDim.x * cuda.blockDim.y),
    ):
        partial_norm_sum += shared_vector[feature_idx] * shared_vector[feature_idx]

    cuda.atomic.add(shared_vector_norm, 0, partial_norm_sum)
    cuda.syncthreads()

    vector_norm = math.sqrt(shared_vector_norm[0])

    for row_idx in range(row_grid_position, num_rows, row_grid_size):
        for column_idx in range(column_grid_position, num_columns, column_grid_size):
            codebook_norm = types.float32(0.0)
            dot_product = types.float32(0.0)

            for feature_idx in range(num_features):
                codebook_value = codebook[row_idx, column_idx, feature_idx]
                codebook_norm += codebook_value * codebook_value
                dot_product += codebook_value * shared_vector[feature_idx]

            distance_matrix[row_idx, column_idx] = 1 - (
                dot_product / (math.sqrt(codebook_norm) * vector_norm)
            )


def _dot(codebook: Codebook, input_vector: Vector, distance_matrix: MapValues):
    row_grid_position, column_grid_position = cuda.grid(2)
    row_grid_size, column_grid_size = cuda.gridsize(2)
    num_rows, num_columns, num_features = codebook.shape

    shared_vector = cuda.shared.array(2048, dtype=types.float32)
    for feature_idx in range(
        (cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x),
        num_features,
        (cuda.blockDim.x * cuda.blockDim.y),
    ):
        shared_vector[feature_idx] = input_vector[feature_idx]
    cuda.syncthreads()

    for row_idx in range(row_grid_position, num_rows, row_grid_size):
        for column_idx in range(column_grid_position, num_columns, column_grid_size):
            dot_product_sum = types.float32(0.0)
            for feature_idx in range(num_features):
                dot_product_sum += (
                    codebook[row_idx, column_idx, feature_idx]
                    * shared_vector[feature_idx]
                )
            distance_matrix[row_idx, column_idx] = dot_product_sum


class CudaDistanceFactory:
    standard_kernels = [
        ("manhattan", _manhattan),
        ("euclidean", _euclidean),
        ("cosine", _cosine),
        ("dot", _dot),
    ]

    def __init__(
        self,
        fastmath: bool = True,
        inline: bool = True,
        lineinfo: bool = False,
        debug_mode: bool = False,
        max_registers: Optional[int] = None,
    ):
        self.fastmath = fastmath
        self.inline = inline
        self.lineinfo = lineinfo
        self.debug = debug_mode
        self.opt = not debug_mode
        self.max_registers = max_registers
        self._kernels: dict[str, CUDADispatcher] = {}

        for name, kernel_func in self.standard_kernels:
            self.register_kernel(name, kernel_func)

    def register_kernel(self, name: str, kernel_func: GPUDeviceDistance) -> None:
        """Register a raw kernel function under a given name"""
        decorated = cuda.jit(
            fastmath=self.fastmath,
            inline=self.inline,
            lineinfo=self.lineinfo,
            opt=self.opt,
            debug=self.debug,
            max_registers=self.max_registers,
        )(kernel_func)
        self._kernels[name] = decorated

    def profile_kernel(
        self,
        args,
        name: str,
        grid_size: tuple,
        block_size: tuple,
        stream: cuda.stream = cuda.default_stream(),
    ):
        """
        Profiles and executes a CUDA kernel.

        Parameters:
        -----------
        args : tuple
            Arguments to pass to the CUDA kernel.
        name : str
            Name of the kernel to profile and execute.
        grid_size : tuple
            Grid size for the CUDA kernel execution.
        block_size : tuple
            Block size for the CUDA kernel execution.

        Raises:
        -------
        ValueError
            If the specified kernel name is not found in the kernel dictionary.
        """

        try:
            kernel = self._kernels[name]
        except KeyError:
            raise ValueError(
                f"Kernel {name} not found. Available kernels: {list(self._kernels.keys())}"
            )

        stream.synchronize()
        with cuda.profiling():
            kernel[grid_size, block_size, stream](*args)

    def get_kernel(self, name: str) -> VectorDistanceFunction:
        """Get a compiled kernel by name"""
        try:
            return self._kernels[name]
        except KeyError:
            raise ValueError(f"Vector distance kernel {name} not found!")
