from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

# from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numpy.typing import NDArray

from chisom._core.cpu.distance import make_map_distance_function
from chisom._core.cpu.kernel import make_kernel_coefficient_function
from chisom._core.types import Codebook


class Trainer(ABC):
    """
    Abstract base class for a trainer that handles the training and prediction of a model using a codebook.

    Parameters
    ----------
    codebook : Codebook
        The codebook used for training and prediction.
    num_rows : int
        Number of rows in the codebook.
    num_columns : int
        Number of columns in the codebook.
    num_features : int
        Number of features in the codebook.

    """

    @abstractmethod
    def __init__(
        self,
        codebook: NDArray,
        map_distance: str,
        kernel_shape: str,
        fastmath: bool,
    ) -> None:
        self.num_rows = codebook.shape[0]
        self.num_columns = codebook.shape[1]
        self.num_features = codebook.shape[2]
        self.num_neurons = self.num_rows * self.num_columns
        self.grid_size = np.array([self.num_rows, self.num_columns], dtype=np.int32)

        self.map_distance_function = make_map_distance_function(map_distance, fastmath)
        self.compute_coefficients = make_kernel_coefficient_function(
            kernel_shape, fastmath
        )

        self.distance_grid = self.map_distance_function(
            np.array([0, 0], dtype=np.int32), self.grid_size
        )

    @abstractmethod
    def train(self, batch: NDArray) -> None:
        """
        Train the model with a batch of data.

        Parameters
        ----------s
        batch : NDArray
            A batch of vectors to train the model on.
        """
        pass

    # Use getter and setter for codebook to ensure codebook is synchronized from GPU to CPU
    @property
    @abstractmethod
    def codebook(self) -> Codebook:
        """Returns the codebook used for training and prediction."""
        pass

    @codebook.setter
    @abstractmethod
    def codebook(self, codebook: Codebook) -> None:
        pass

    @abstractmethod
    def predict(self, batch: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Predicts the best matching unit (BMU) and quantization error (QE) for each vector in the batch.

        Parameters
        ----------
        batch : NDArray
            A batch of vectors to predict BMU and QE for.

        Returns
        -------
        Tuple[NDArray, NDArray]
            A tuple containing:
            - bmus_out: An array of indices of the best matching units for each vector.
            - qe_out: An array of quantization errors for each vector.
        """
        pass

    def update_coefficients(
        self,
        alpha: np.float32,
        sigma: np.float32,
        epoch: np.int32,
    ) -> None:
        """
        Update the coefficients used in training.
        This enables calulation of distances and dependent values for the gaussion neighborhood only once per epoch.

        Parameters
        ----------
        alpha : np.float32
            Learning rate.
        sigma : np.float32
            Neighborhood radius.
        sigma_sub : np.float32
            A substitute for the sigma value, used in local neighbourhood calculation to ensure the update value is 0 at the end of the radius.
        epoch : np.int32
            Epoch number, used to calculate the coefficients.
        """
        self.alpha = alpha
        self.sigma = sigma
        self.epoch = epoch

        self.computed_coefficients = self.compute_coefficients(
            self.distance_grid, self.alpha, self.sigma
        )

    @property
    @abstractmethod
    def target(self) -> str:
        """
        Returns the target device for the trainer.

        Returns
        -------
        str
            The target device, e.g., 'cpu' or 'cuda'.
        """
        pass
