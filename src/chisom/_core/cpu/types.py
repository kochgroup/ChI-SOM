from typing import Annotated, Callable, Tuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

from chisom._core.types import Codebook, MapValues, Position, Vector

Neighbors: TypeAlias = Annotated[NDArray[np.int32], "int32[:,::1]"]

VectorDistanceFunction: TypeAlias = Callable[
    [Codebook, Vector],
    MapValues,
]

CodebookUpdateFunction: TypeAlias = Callable[
    [Codebook, Vector, MapValues, Vector],
    None,
]

LocalUpdateFunction: TypeAlias = Callable[
    [Codebook, Vector, Neighbors, Vector],
    None,
]

BroadcastingDistanceFunction: TypeAlias = Callable[
    [
        Annotated[
            NDArray[np.float32], "float32[::1], float32[:,::1], float32[:,:,::1]"
        ],
        Annotated[
            NDArray[np.float32], "float32[::1], float32[:,::1], float32[:,:,::1]"
        ],
    ],
    NDArray[np.float32],
]

MapDistanceFunction: TypeAlias = Callable[
    [Position, Position],
    MapValues,
]

BoundedDistanceFunction: TypeAlias = Callable[[np.float32], Tuple[Vector, Vector]]

PairwiseDistanceFunction: TypeAlias = Callable[[MapValues, Position], MapValues]

CodebookCoefficientsFunction: TypeAlias = Callable[
    [MapValues, np.float32, np.float32],
    MapValues,
]
