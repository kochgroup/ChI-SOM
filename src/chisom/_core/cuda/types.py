from typing import Annotated, Callable, TypeAlias

from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba.cuda.dispatcher import CUDADispatcher

GPUCodebook: TypeAlias = Annotated[DeviceNDArray, "float32[::1,:,:]"]
GPUMapValues: TypeAlias = Annotated[DeviceNDArray, "float32[::1, :1]"]
GPUVector: TypeAlias = Annotated[DeviceNDArray, "int32[::1]"]
GPUPosition: TypeAlias = Annotated[DeviceNDArray, "int32[::1]"]

VectorDistanceFunction: TypeAlias = Callable[
    [GPUCodebook, GPUVector, GPUPosition],
    None,
]

CodebookUpdateFunction: TypeAlias = Callable[
    [GPUCodebook, GPUVector, GPUMapValues, GPUVector],
    None,
]

GPUDeviceDistance: TypeAlias = Annotated[CUDADispatcher, "VectorDistanceFunction"]
GPUDeviceUpdate: TypeAlias = Annotated[CUDADispatcher, "CodebookUpdateFunction"]
