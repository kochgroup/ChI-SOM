#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from chisom._core.cpu import trainer as cpu_trainer
from chisom._core.cpu.umatrix import make_umatrix_calculation
from chisom._core.types import Codebook, UMatrix
from chisom.io._types import DataLoader
from chisom.io._utils import numpy_collate

try:
    from chisom._core.cuda import trainer as gpu_trainer

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class Som:
    """Main Class to create and train a Self-Organizing Map"""

    vector_distances = ["manhattan", "euclidean", "cosine"]

    map_distances = [
        "manhattan",
        "euclidean",
        "manhattan_toroid",
        "euclidean_toroid",
    ]

    neighborhood_kernels = ["gaussian", "mexican", "cone"]

    def __init__(
        self,
        rows: int,
        columns: int,
        features: int,
        vector_distance: str = "euclidean",
        map_distance: str = "euclidean_toroid",
        neighborhood_kernel: str = "gaussian",
        use_cuda: bool = False,
        use_local_neighborhood: bool = False,
        use_fastmath: bool = True,
        save_progress: Optional[str] = None,
        low: float = 0.0,
        high: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initializes the Som Object.

        Parameters
        ----------
        rows
            Number of rows of neurons.
        columns
            Number of columns of neurons.
        features
            Numbers of features to the data / weights of each neuron.
        vector_distance
            Distance used in original data space, by default "euclidean".
            Possible values: "euclidean", "manhattan", "cosine"
        map_distance
            Distance used in map space, by default "euclidean_toroid".
            Possbile values: "euclidean", "manhattan", "euclidean_toroid", "manhattan_toroid"
        neighborhood_kernel
            Shape of the neighborhood kernel, by default "gaussian".
        use_cuda
            If True, CUDA accelleration is used. Needs numba-cuda. By default False.
        use_local_neighborhood
            Sets a hard neighborhood cutoff, by default False.
            Only used on CPU. Significantly increases performance at cost of numerical accuracy.
        use_fastmath
            Slightly decrease numerical accuracy to increase performance, by default True.
        save_progress
            Saves codebook and U-Matrix to the given location if set, by default None.
            Usefull if long running computations crash / time out.
        low
            Lower bound for codebook initialization, by default 0.0.
        high
            Upper bound for codebook initialization, by default 1.0.
        seed
            Randomness seed for replicability, by default None.

        Raises
        ------
        ValueError
            If the map dimensions are less than 1x1.
        ValueError
            If the number of features is less than 2.
        ImportError
            If CUDA is requested but not available.
        ValueError
            If the vector distance norm is not one of the supported norms.
        ValueError
            If the map distance norm is not one of the supported norms.
        ValueError
            If the neighborhood kernel is not one of the supported kernels.
        """

        # Sanity checks and definitions
        if rows <= 0 or columns <= 0:
            raise ValueError("Map dimension must be at least 1x1")
        else:
            self.rows = rows
            self.columns = columns
        if features <= 1:
            raise ValueError("Needs at least 2 features")
        else:
            self.features = features

        if use_cuda and not CUDA_AVAILABLE:
            raise ImportError(
                "CUDA is not available. Please install the CUDA version of chi-som."
            )

        self.use_cuda = use_cuda and CUDA_AVAILABLE
        self.use_local_neighborhood = use_local_neighborhood
        self.fastmath = use_fastmath

        if save_progress is not None:
            self.outpath: Path | None = Path(save_progress)
            self.outpath.mkdir(parents=False, exist_ok=False)
        else:
            self.outpath = None

        self.save_progress = save_progress

        if vector_distance not in Som.vector_distances:
            raise ValueError(f"Unknown vector distance norm: {vector_distance}")
        else:
            self.vector_distance_norm = vector_distance

        if map_distance not in Som.map_distances:
            raise ValueError(f"Unknown map distance norm: {map_distance}")
        else:
            self.map_distance_norm = map_distance

        if neighborhood_kernel not in Som.neighborhood_kernels:
            raise ValueError(f"Unknown neighborhood kernel: {neighborhood_kernel}")
        else:
            self.neighborhood_kernel = neighborhood_kernel

        # Initial setup of codebook
        self.dimensions = np.array((self.rows, self.columns), dtype=np.int32)
        self.rng = np.random.default_rng(seed)

        codebook = self.rng.uniform(low, high, (self.rows, self.columns, self.features))
        codebook = np.asarray(codebook, dtype=np.float32, order="C")
        trainer_type: type

        if self.use_cuda:
            trainer_type = gpu_trainer.CudaTrainer
        else:
            if self.use_local_neighborhood:
                trainer_type = cpu_trainer.CPUTrainerLocal
            else:
                trainer_type = cpu_trainer.CPUTrainer

        self.trainer_instance = trainer_type(
            codebook,
            self.vector_distance_norm,
            self.map_distance_norm,
            self.neighborhood_kernel,
            self.fastmath,
        )

        self.umatrix: UMatrix

    def train(
        self,
        data: NDArray | DataLoader,
        epoch: int,
        sigma: float,
        alpha: float,
    ):
        """
        Train the SOM with the given data for one epoch

        Parameters
        ----------
        data : NDArray | DataLoader
            The data to train the SOM with. If a DataLoader is used, it should be batched.
            If a numpy array is used, it will be treated as a single batch.
        epoch : int
            The current epoch of the training.
        sigma : float
            The sigma value for the current epoch.
            This is used to calculate the neighborhood radius.
            Must be greater than 0.
        alpha : float, optional
            The learning rate for the current epoch.

        Raises
        ------
        ValueError
            If sigma is less than or equal to 0.
        """

        if sigma <= 0:
            raise ValueError("Sigma can not be zero or smaller")

        # Transform the input data to adhere to batching and CPU training if necessary
        data = self._transform_in_data(data)

        # Use the update coefficients function to set the trainers parameter for the epoch, including factors for map distance
        self.trainer_instance.update_coefficients(
            alpha=np.float32(alpha),
            sigma=np.float32(sigma),
            epoch=np.int32(epoch),
        )
        for batch in data:
            self.trainer_instance.train(batch)

        # Save the umatrix and codebook if save_progress is set
        if self.outpath is not None:
            self.umatrix = np.concat(
                (
                    self.umatrix,
                    self.get_umatrix()[np.newaxis, :, :],
                ),
                axis=0,
            )
            np.save(self.outpath / "umatrix", self.umatrix)
            np.save(self.outpath / "codebook", self.trainer_instance.codebook)

    @property
    def codebook(self) -> Codebook:
        return self.trainer_instance.codebook

    @codebook.setter
    def codebook(self, codebook: Codebook) -> None:
        self.trainer_instance.codebook = codebook

    def get_umatrix(self) -> UMatrix:
        """
        Calculate the UMatrix for the current codebook

        Returns
        -------
        UMatrix
            The UMatrix for the current codebook.
        """
        # TODO move to trainer factory, support more distances
        umatrix_func = make_umatrix_calculation(self.vector_distance_norm)
        umatrix = umatrix_func(self.codebook)

        return umatrix

    def predict(
        self, data: NDArray | DataLoader
    ) -> Tuple[NDArray[np.uint16], NDArray[np.float32]]:
        """
        Return the positions of the BMU for a dataset

        Parameters
        ----------
        data
            Dataset to find the BMUs for.

        Returns
        -------
        NDArray[np.uint16]
            The BMUs for the data.
        NDArray[np.float32]
            The Quantization Error

        Raises
        ------
        TypeError
            Error if the data format is not known
        """

        data = self._transform_in_data(data)
        if isinstance(data, DataLoader):
            data.shuffle = False

        bmu_batches, qe_batches = [], []
        for batch in data:
            bmu_batch, qe_batch = self.trainer_instance.predict(batch)
            bmu_batches.append(bmu_batch)
            qe_batches.append(qe_batch.flatten())
        bmu = np.vstack(bmu_batches)
        qe = np.concat(qe_batches)

        return bmu.astype(np.uint16), qe

    def _transform_in_data(
        self, data: NDArray | DataLoader
    ) -> NDArray[np.float32] | DataLoader:
        # Ad new dimension if a numpy array is used as the input format,
        # to conform to batched data iterating
        if isinstance(data, np.ndarray):
            data = np.astype(data[np.newaxis, :], np.float32)

        # return_fp_from_dict is necessary to select to correct column from the Dataloader
        if isinstance(data, DataLoader):
            # Collate to numpy arrays if using CPU calculation
            if not self.use_cuda:
                data.collate_fn = numpy_collate

        return data
