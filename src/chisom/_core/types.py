from typing import Annotated, TypeAlias

import numpy as np
from numpy.typing import NDArray

Codebook: TypeAlias = Annotated[NDArray[np.float32], "float32[:, :, ::1]"]
Vector: TypeAlias = Annotated[NDArray[np.float32], "float32[::1]"]
Position: TypeAlias = Annotated[NDArray[np.int32], "int32[::1]"]
MapValues: TypeAlias = Annotated[NDArray[np.float32], "float32[:, ::1]"]
UMatrix: TypeAlias = Annotated[NDArray[np.float16], "float16[:, ::1]"]
