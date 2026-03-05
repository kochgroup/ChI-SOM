from collections.abc import Callable
from typing import Dict, List, Set, Tuple, TypeVar

import numpy as np
import tables
import zarr
from numpy.typing import NDArray

Mol = TypeVar("Mol")
Smiles = TypeVar("Smiles", bound=str)
Atom_Type = TypeVar("Atom_Type", str, int, float, np.dtype)
MolGenerator = Callable[[str], Mol]
FileList = Dict[str, list[str]]
ExtraColumns = Dict[str, Tuple[int, str]]
InputLine = List[str]
OutputLine = Dict[str, List[str] | NDArray]
LeafMap = Dict[str, Tuple[int, Atom_Type, str] | Tuple[int, Atom_Type]]
Range = TypeVar("Range", List[float], Set)
RangesDict = Dict[str, Dict[str, str | Range]]
FileRoot = TypeVar("FileRoot", tables.Group, zarr.Group)
FingerprintStack = TypeVar("FingerprintStack", NDArray, List[NDArray])
Packer = Callable[[FingerprintStack], NDArray]


class rdFingerprintGenerator:
    def __init__(self): ...
    def GetFingerprintAsNumPy(self, mol: Mol) -> NDArray[np.uint8]: ...


class DataLoader:
    collate_fn: Callable

    def __init__(
        self, dataset, batch_size, shuffel, num_workers, collate_fn, pin_memory
    ): ...


Timeout = TypeVar("Timeout", int, float)
Message = TypeVar("Message", InputLine, OutputLine)
