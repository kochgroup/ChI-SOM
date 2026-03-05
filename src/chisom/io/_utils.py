"""Contains a copy of the pytorch torch.uitls.data._utils.collate function with some modifications."""

import collections
import copy
import re
from typing import Callable, Dict, Optional, Tuple, Type, Union

import numpy as np

np_str_obj_array_pattern = re.compile(r"[SaUO]")

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def collate(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    """
    General collate function that handles collection type of element within each batch.

    The function also opens function registry to deal with specific element types. `default_collate_fn_map`
    provides default collate functions for tensors, numpy arrays, numbers and strings.

    Args:
        batch: a single batch to be collated
        collate_fn_map: Optional dictionary mapping from element type to the corresponding collate function.
            If the element type isn't present in this dictionary,
            this function will go through each key of the dictionary in the insertion order to
            invoke the corresponding collate function if the element type is a subclass of the key.

    Examples:
        >>> def collate_tensor_fn(batch, *, collate_fn_map):
        ...     # Extend this function to handle batch of tensors
        ...     return torch.stack(batch, 0)
        >>> def custom_collate(batch):
        ...     collate_map = {torch.Tensor: collate_tensor_fn}
        ...     return collate(batch, collate_fn_map=collate_map)
        >>> # Extend `default_collate` by in-place modifying `default_collate_fn_map`
        >>> default_collate_fn_map.update({torch.Tensor: collate_tensor_fn})

    Note:
        Each collate function requires a positional argument for batch and a keyword argument
        for the dictionary of collate functions as `collate_fn_map`.
    """
    elem = batch[0]
    elem_type = type(elem)

    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](
                    batch, collate_fn_map=collate_fn_map
                )

    if isinstance(elem, collections.abc.Mapping):
        try:
            if isinstance(elem, collections.abc.MutableMapping):
                # The mapping type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new mapping.
                # Create a clone and update it if the mapping type is mutable.
                clone = copy.copy(elem)
                clone.update(
                    {
                        key: collate(
                            [d[key] for d in batch], collate_fn_map=collate_fn_map
                        )
                        for key in elem
                    }
                )
                return clone
            else:
                return elem_type(
                    {
                        key: collate(
                            [d[key] for d in batch], collate_fn_map=collate_fn_map
                        )
                        for key in elem
                    }
                )
        except TypeError:
            # The mapping type may not support `copy()` / `update(mapping)`
            # or `__init__(iterable)`.
            return {
                key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map)
                for key in elem
            }
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(
            *(
                collate(samples, collate_fn_map=collate_fn_map)
                for samples in zip(*batch)
            )
        )
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [
                collate(samples, collate_fn_map=collate_fn_map)
                for samples in transposed
            ]  # Backwards compatibility.
        else:
            try:
                if isinstance(elem, collections.abc.MutableSequence):
                    # The sequence type may have extra properties, so we can't just
                    # use `type(data)(...)` to create the new sequence.
                    # Create a clone and update it if the sequence type is mutable.
                    clone = copy.copy(elem)  # type: ignore[arg-type]
                    for i, samples in enumerate(transposed):
                        clone[i] = collate(samples, collate_fn_map=collate_fn_map)
                    return clone
                else:
                    return elem_type(
                        [
                            collate(samples, collate_fn_map=collate_fn_map)
                            for samples in transposed
                        ]
                    )
            except TypeError:
                # The sequence type may not support `copy()` / `__setitem__(index, item)`
                # or `__init__(iterable)` (e.g., `range`).
                return [
                    collate(samples, collate_fn_map=collate_fn_map)
                    for samples in transposed
                ]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def collate_numpy_array_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    elem = batch[0]
    # array of string classes and object
    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
        raise TypeError(default_collate_err_msg_format.format(elem.dtype))

    return np.stack(batch, 0).astype(np.float32)


def collate_numpy_scalar_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    return np.asarray(batch).astype(np.float32)


def collate_float_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    return np.array(batch, dtype=np.float32)


def collate_int_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    return np.array(batch)


def collate_str_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    return batch


numpy_collate_fn_map: Dict[Union[Type, Tuple[Type, ...]], Callable] = {
    np.ndarray: collate_numpy_array_fn
}
numpy_collate_fn_map[(np.bool_, np.number, np.object_)] = collate_numpy_scalar_fn
numpy_collate_fn_map[float] = collate_float_fn
numpy_collate_fn_map[int] = collate_int_fn
numpy_collate_fn_map[str] = collate_str_fn
numpy_collate_fn_map[bytes] = collate_str_fn


def numpy_collate(batch):
    """
    Modified copy of the pytorch torch.utils.data._utils.collate function, returning a collated batch of numpy arrays instead of torch tensors.
    """
    return collate(batch, collate_fn_map=numpy_collate_fn_map)
