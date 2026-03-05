from abc import ABC
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from chisom._core.base import Trainer
from chisom._core.cpu.distance import (
    make_bounded_distance_func,
    make_vector_distance_func,
)
from chisom._core.cpu.kernel import (
    make_local_kernel_coefficient_function,
    make_local_update_function,
    make_update_function,
)
from chisom._core.types import Codebook

FASTMATH_FLAG = True


class CPUTrainerBase(Trainer, ABC):
    def __init__(
        self,
        codebook,
        vector_distance_norm: str,
        map_distance_norm: str,
        kernel_shape: str,
        fastmath: bool = FASTMATH_FLAG,
    ):
        super().__init__(
            codebook=codebook,
            map_distance=map_distance_norm,
            kernel_shape=kernel_shape,
            fastmath=fastmath,
        )
        self.vector_distance = make_vector_distance_func(vector_distance_norm, fastmath)
        self.codebook = codebook

    @property
    def codebook(self) -> Codebook:
        return self._codebook

    @codebook.setter
    def codebook(self, codebook: Codebook):
        self._codebook = codebook

    @property
    def target(self):
        return "cpu"

    def predict(self, batch: NDArray) -> Tuple[NDArray, NDArray]:
        bmus = []
        qe = []
        for vector in batch:
            dist = self.vector_distance(self._codebook, vector)
            bmus.append(dist.argmin())
            qe.append(dist.min())

        bmus_out = np.asarray(
            np.column_stack(np.unravel_index(bmus, self.grid_size)), dtype=np.int32
        )
        qe_out = np.asarray(qe, dtype=np.float32)
        return bmus_out, qe_out


class CPUTrainerLocal(CPUTrainerBase):
    def __init__(
        self,
        codebook,
        vector_distance_norm,
        map_distance_norm,
        kernel_shape,
        fastmath=FASTMATH_FLAG,
    ):
        self.map_distance_func = make_bounded_distance_func(map_distance_norm, fastmath)
        self.update_function = make_local_update_function(fastmath)
        super().__init__(
            codebook=codebook,
            vector_distance_norm=vector_distance_norm,
            map_distance_norm=map_distance_norm,
            kernel_shape=kernel_shape,
        )
        self.compute_coefficients = make_local_kernel_coefficient_function(
            kernel_shape, fastmath
        )

    def train(self, batch):
        for vector in batch:
            bmu = self.vector_distance(self._codebook, vector).argmin()
            bmu_row, bmu_col = np.divmod(bmu, self.grid_size[1])
            bmu_pos = np.array([bmu_row, bmu_col], dtype=np.int32)
            neighbors = np.mod((self.relative_neighbors + bmu_pos), self.grid_size)
            self.update_function(
                self._codebook,
                vector,
                neighbors,
                self.computed_coefficients,
            )

    def update_coefficients(self, alpha, sigma, epoch):
        self.alpha = alpha
        self.sigma = sigma
        self.relative_neighbors, relative_distances = self.map_distance_func(sigma)
        if epoch == 0:
            if self.num_columns > self.num_rows:
                self.relative_neighbors = np.delete(self.relative_neighbors, -1, 0)
                relative_distances = np.delete(relative_distances, -1, 0)

        self.computed_coefficients = self.compute_coefficients(
            relative_distances,
            alpha,
            sigma,
        )


class CPUTrainer(CPUTrainerBase):
    def __init__(
        self,
        codebook,
        vector_distance_norm,
        map_distance_norm,
        kernel_shape,
        fastmath=FASTMATH_FLAG,
        **kwargs,
    ):
        self.update_function = make_update_function(fastmath)
        super().__init__(
            codebook=codebook,
            vector_distance_norm=vector_distance_norm,
            map_distance_norm=map_distance_norm,
            kernel_shape=kernel_shape,
        )

    def train(self, batch):
        for vector in batch:
            bmu = self.vector_distance(self._codebook, vector).argmin()
            bmu_row, bmu_col = np.divmod(bmu, self.grid_size[1])
            bmu_pos = np.array([bmu_row, bmu_col], dtype=np.int32)
            self.update_function(
                self._codebook, vector, self.computed_coefficients, bmu_pos
            )
