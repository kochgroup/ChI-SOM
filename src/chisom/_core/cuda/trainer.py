import warnings
from typing import Tuple

import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
from numba.types import float32, int32
from numpy.typing import NDArray

from chisom._core.base import Trainer
from chisom._core.cuda.distance import CudaDistanceFactory
from chisom._core.cuda.types import CodebookUpdateFunction
from chisom._core.cuda.utils import cuda_argmin_2d
from chisom._core.types import Codebook, MapValues, Position, Vector

FASTMATH_FLAG = True

COMPUTE_CAPABILITY_MAPPING = {
    "default": {
        "TBP": (32, 16),
        "max_registers": 32,
    },
    "(8, 9)": {
        "TBP": (32, 16),
        "max_registers": 40,
    },
    "(9, 0)": {
        "TBP": (32, 24),
        "max_registers": 32,
    },
}


def make_update_function(
    fastmath_flag: bool = FASTMATH_FLAG, lineinfo: bool = False, max_registers: int = 0
) -> CodebookUpdateFunction:
    @cuda.jit(
        fastmath=fastmath_flag,
        lineinfo=lineinfo,
        max_registers=max_registers,
    )
    def update_codebook(
        codebook: Codebook,
        vector: Vector,
        coefficients: MapValues,
        bmu_position: Position,
    ) -> None:
        row_grid_position, column_grid_position = cuda.grid(2)
        row_grid_size, column_grid_size = cuda.gridsize(2)
        num_rows, num_cols, num_features = codebook.shape

        shared_bmu_position = cuda.shared.array(2, dtype=int32)
        if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
            shared_bmu_position[0] = bmu_position[0]
            shared_bmu_position[1] = bmu_position[1]

        shared_vector = cuda.shared.array(2048, dtype=float32)
        # threadIdx.x is the fasted changing, so needs to be coalesced. n_features > (cuda.blockDim.x + cuda.blockDim.y) is handled.
        for feature_idx in range(
            (cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x),
            num_features,
            (cuda.blockDim.x * cuda.blockDim.y),
        ):
            shared_vector[feature_idx] = vector[feature_idx]
        cuda.syncthreads()

        for row_idx in range(row_grid_position, num_rows, row_grid_size):
            for column_id in range(column_grid_position, num_cols, column_grid_size):
                offset_row = abs(shared_bmu_position[0] - row_idx)
                offset_column = abs(shared_bmu_position[1] - column_id)
                coefficient = coefficients[offset_row, offset_column]

                for feature_idx in range(num_features):
                    value = coefficient * (
                        shared_vector[feature_idx]
                        - codebook[row_idx, column_id, feature_idx]
                    )
                    codebook[row_idx, column_id, feature_idx] += value

    return update_codebook


class CudaTrainer(Trainer):
    """
    A CUDA-based trainer for Self-Organizing Maps (SOMs).

    This class provides a CUDA-accelerated implementation of the SOM training
    algorithm. It uses Numba's CUDA Python API to launch GPU kernels for
    computing distances and updating the codebook, significantly improving
    training performance compared to CPU implementations.

    Parameters
    ----------
    codebook : numpy.ndarray
        The initial codebook for the SOM, shape (n_rows, n_columns, n_features).
    vector_distance_norm : str
        The norm to use for computing distances between vectors.
        Options: 'manhattan', 'euclidean', 'cosine', 'dot'
    map_distance_norm : str
        The norm to use for computing distances between map units.
        Options: 'manhattan', 'euclidean'
    kernel_shape : str
        The type of kernel to use for updating the codebook.
        Options: 'gaussian', 'mexican_hat', 'cone'
    fastmath : bool, optional
        Whether to use fast math operations for CUDA kernels (default is True).
        Enables faster but potentially less accurate floating-point operations.

    Attributes
    ----------
    num_rows : int
        The number of rows in the codebook.
    num_columns : int
        The number of columns in the codebook.
    num_neurons : int
        The total number of neurons in the codebook (num_rows * num_columns).
    num_features : int
        The number of features in each neuron.


    vector_distance_function : callable
        The compiled CUDA kernel for computing distances between vectors.

    Methods
    -------
    train(batch)
        Trains the SOM using the given batch of input vectors.
    predict(batch)
        Finds the best matching units (BMUs) for the given batch of input vectors.
    update_coefficients(alpha, sigma, sigma_sub, epoch)
        Updates the neighborhood coefficients based on the current training parameters.
    codebook
        Property that returns the current codebook as a NumPy array.
    target
        Property that returns the target computing device ("cuda").
    """

    def __init__(
        self,
        codebook,
        vector_distance_norm,
        map_distance_norm,
        kernel_shape,
        fastmath=FASTMATH_FLAG,
        *args,
        **kwargs,
    ):
        super().__init__(
            codebook=codebook,
            map_distance=map_distance_norm,
            kernel_shape=kernel_shape,
            fastmath=fastmath,
        )
        self.compute_capablity = str(cuda.get_current_device().compute_capability)
        self._cuda_stream = cuda.stream()

        # Use perfomance tested threads per block and max registers, if available
        if self.compute_capablity in COMPUTE_CAPABILITY_MAPPING.keys():
            self.threads_per_block = COMPUTE_CAPABILITY_MAPPING[self.compute_capablity][
                "TBP"
            ]
            self.max_registers = COMPUTE_CAPABILITY_MAPPING[self.compute_capablity][
                "max_registers"
            ]
        else:
            self.threads_per_block = COMPUTE_CAPABILITY_MAPPING["default"]["TBP"]
            self.max_registers = COMPUTE_CAPABILITY_MAPPING["default"]["max_registers"]

        row_blocks_to_calculate = (self.num_rows // self.threads_per_block[0]) + 1
        column_blocks_to_calculate = (self.num_columns // self.threads_per_block[1]) + 1

        row_blocks = max(row_blocks_to_calculate, 32)
        column_blocks = max(column_blocks_to_calculate, 48)

        try:
            assert row_blocks * column_blocks <= (2**31 - 1), (
                "Too many blocks for a cuda grid"
            )
        except AssertionError:
            raise ValueError("Too many blocks for a cuda grid. Manually set the TBP")

        self.blocks_per_grid = (row_blocks, column_blocks)

        self.codebook = codebook

        self.vector_distance_factory = CudaDistanceFactory(
            fastmath=fastmath,
        )
        self.vector_distance_function = self.vector_distance_factory.get_kernel(
            vector_distance_norm
        )

        self.update_function = make_update_function(fastmath)

        self.codebook_vector_distance = cuda.device_array(
            (self.num_rows, self.num_columns),
            dtype=np.float32,
            order="F",
            stream=self._cuda_stream,
        )
        self.partial_argmin = cuda.device_array(
            (row_blocks, column_blocks, 2),
            np.int32,
            stream=self._cuda_stream,
        )
        self.array_size = cuda.mapped_array(
            2, dtype=np.float32, stream=self._cuda_stream
        )

        warnings.simplefilter("ignore", NumbaPerformanceWarning)

    @property
    def target(self):
        return "cuda"

    def train(self, batch):
        device_batch = cuda.to_device(batch, stream=self._cuda_stream)
        for vector in device_batch:
            cuda.synchronize()

            self.vector_distance_function[
                self.blocks_per_grid,
                self.threads_per_block,
                self._cuda_stream,
            ](
                self._codebook,
                vector,
                self.codebook_vector_distance,
            )

            argmin = cuda_argmin_2d(
                self.codebook_vector_distance,
                self.partial_argmin,
                self.array_size,
                self._cuda_stream,
            )

            self.update_function[
                self.blocks_per_grid,
                self.threads_per_block,
                self._cuda_stream,
            ](
                self._codebook,
                vector,
                self.device_coefficients,
                argmin,
            )

    @property
    def codebook(self):
        cuda.synchronize()
        cpu_codebook = self._codebook.copy_to_host(stream=self._cuda_stream)
        return np.asarray(cpu_codebook, order="C")

    @codebook.setter
    def codebook(self, codebook: NDArray):
        self._codebook = cuda.to_device(
            np.array(
                codebook,
                dtype=np.float32,
                order="F",
            ),
            stream=self._cuda_stream,
        )
        cuda.synchronize()

    def predict(self, batch: NDArray) -> Tuple[NDArray, NDArray]:
        bmu = []
        qe = []
        d_batch = cuda.to_device(batch, stream=self._cuda_stream)
        with cuda.defer_cleanup():
            for vector in d_batch:
                self.vector_distance_function[
                    self.blocks_per_grid,
                    self.threads_per_block,
                    self._cuda_stream,
                ](
                    self._codebook,
                    vector,
                    self.codebook_vector_distance,
                )

                argmin = cuda_argmin_2d(
                    self.codebook_vector_distance,
                    self.partial_argmin,
                    self.array_size,
                    self._cuda_stream,
                )

                bmu.append(argmin.copy_to_host())
                qe.append(self.codebook_vector_distance[argmin[0], argmin[1]])
        return (np.vstack(bmu), np.stack(qe))

    def update_coefficients(self, alpha, sigma, epoch):
        super().update_coefficients(alpha, sigma, epoch)
        self.device_coefficients = cuda.to_device(
            self.computed_coefficients.astype(np.float32, order="F"), self._cuda_stream
        )

    def __del__(self):
        del self._codebook
        del self.codebook_vector_distance
        del self.partial_argmin
        del self.array_size
        del self._cuda_stream
