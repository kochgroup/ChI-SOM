import fileinput
import multiprocessing as mp
import pathlib as pl
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sized
from math import ceil
from queue import Queue, ShutDown

import numpy as np
import tables
from tqdm import tqdm

from chisom.io._types import FileList, FileRoot, InputLine, LeafMap, RangesDict
from chisom.io.datastore_factories import DataloaderFingerprintGeneratorFactory

# FIXME Does not work with multiprocessing spawn (win, mac)
# For this, encapsulation for parsers/generator functions for pickeling is needed

# Fitler out repetitive warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The codec `vlen-utf8`")
warnings.filterwarnings(
    "ignore", category=UserWarning, message="The dtype `StringDType()`"
)


class LazyIterator(Iterator, Sized):
    """
    Lazy iterator for reading lines from a file object in batches. Needed as only Iterables of denfined length can be used in multiprocessing.
    """

    def __init__(
        self,
        file_object: fileinput.FileInput,
        estimated_len: int,
        delimiter: str = ",",
        batch_size: int = 1,
        skip_lines: int = 0,
    ):
        self.__len = ceil(estimated_len / batch_size)
        self.fileObject = file_object
        self.delimiter = delimiter
        self.batch_size = batch_size
        self.stop_iter_flag = False
        self.skip_lines = skip_lines

    def _lineskip(self, line: InputLine) -> InputLine:
        for _ in range(self.skip_lines):
            line = next(self.fileObject)
        return line

    def __len__(self) -> int:
        return self.__len

    def __iter__(self):
        return self

    def __next__(self) -> list[InputLine]:
        buffer: list[InputLine] = []
        if not self.stop_iter_flag:
            for _ in range(self.batch_size):
                try:
                    line = next(self.fileObject)
                    if self.fileObject.isfirstline():
                        line = self._lineskip(line)

                except StopIteration:
                    self.stop_iter_flag = True
                    break
                except Exception:
                    ...
                else:
                    line = line.strip()
                    parts = line.split(self.delimiter)
                    buffer.append(parts)

            if len(buffer) > 0:
                return buffer
            else:
                raise StopIteration
        else:
            raise StopIteration


def _parse_file_hierarchy(
    file_hierarchy: FileList,
    file_extentions: list[str],
) -> FileList:
    """
    Parse a file hierarchy dictionary to extract paths to files with specified extensions.

    This function processes a dictionary where keys represent classes and values
    represent paths (either files or directories). It traverses directories and
    collects all files with the specified extensions.

    Parameters
    ----------
    file_hierarchy : FileList
        Dictionary mapping class names to lists of file or directory paths.
        Each path can be a file or a directory containing files.
    file_extentions : list[str], optional
        List of file extensions to filter by, by default ["txt"].
        Extensions can be provided with or without the leading dot.

    Returns
    -------
    FileList
        Dictionary with the same keys as the input, but values are lists of
        absolute paths to files matching the specified extensions.

    Raises
    ------
    TypeError
        If the input file_hierarchy is not a dictionary.
    FileNotFoundError
        If no files with the specified extensions are found for a class.

    Notes
    -----
    - For directory paths, the function recursively walks through all subdirectories.
    - Only files with the specified extensions are included in the result.
    - All paths in the returned dictionary are absolute paths.
    """
    file_extentions = [
        f".{ext}" if not ext.startswith(".") else ext for ext in file_extentions
    ]

    if not isinstance(file_hierarchy, dict):
        raise TypeError("Input is not of type Dictionary!")

    parsed_hierarchy = {}
    for k in file_hierarchy:
        current_pathlist = file_hierarchy[k]

        if not isinstance(current_pathlist, list):
            raise TypeError(f"Input node {current_pathlist} is not of type 'list'")

        parsed_path = []

        for entry in current_pathlist:
            try:
                p = pl.Path(entry).absolute()
            except Exception:
                continue

            if p.is_dir():
                for root, dirs, files in p.walk():
                    for f in files:
                        file = root / f
                        if file.suffix in file_extentions:
                            parsed_path.append(str(file.absolute()))

            elif p.is_file() and p.suffix in file_extentions:
                parsed_path.append(str(p.absolute()))

        if len(parsed_path) == 0:
            raise FileNotFoundError(
                f"File with absolute path {p} for class '{k} was not found!"
            )

        parsed_hierarchy[k] = parsed_path

    return parsed_hierarchy


def _parse_output_path(out_path: str, file_extentions: list[str]) -> str:
    try:
        p = pl.Path(out_path)
    except Exception:
        raise TypeError("Output path is not of type string!")

    if p.is_dir():
        out_file = p.absolute() / f"{p.stem}.h5"

    elif p.is_file():
        answer_challenge = input(
            f"File at {p} already exists. Do you want to overwrite? (y/n):\n"
        )

        if answer_challenge.lower() != "y":
            raise FileExistsError(f"File at {p} already exists")
        else:
            out_file = p.absolute()
    else:
        if not p.parent.is_dir():
            raise FileNotFoundError(f"Path {p} does not exist!")

        if not p.suffix:
            out_file = p.absolute() / ".zarr"
        elif p.suffix in file_extentions:
            out_file = p.absolute()
        else:
            raise ValueError(f"Wrong file extention for file with path {p}!")

    return str(out_file)


def estimate_lines(files: list[str]) -> int:
    """
    Estimate number of line in group of files

    Parameters
    ----------
    files : List[str]
        List of files

    Returns
    -------
    int
        Nuber to total estimated lines

    Notes
    -----
    Adapted from J.F.Sebastian @ stackoverflow
    """
    LEARN_SIZE = 2**10

    size_files = sum([pl.Path(file_).stat().st_size for file_ in files])

    buffer_estimates = []
    for i in range(len(files)):
        with open(files[i], "rb") as file:
            buffer = file.read(LEARN_SIZE)
            buffer_estimates.append(len(buffer) // buffer.count(b"\n"))

    return ceil(size_files / (sum(buffer_estimates) / len(buffer_estimates)))


class StoreCreator(ABC):
    """
    Abstract base class for creating stores from file hierarchies.

    This class provides a template for creating stores from file hierarchies.
    It defines the basic structure and methods that subclasses should implement.
    """

    file_list_prototype: FileList = {
        "group_A": [
            "path/to/file_1.smi",
            "path/to/file_2.smi",
        ],
        "group_B": [
            "path/to/directory_1",
            "path/to/directory_2",
        ],
    }
    default_input_extensions: list[str] = [".txt", ".smi", ".csv"]

    def __init__(
        self,
        fingerprint_generator_factory: DataloaderFingerprintGeneratorFactory,
        file_extensions: list[str],
        num_processes: int = (mp.cpu_count() - 2),
        chunk_size: int = 1000,
        queue_size: int = 500,
    ):
        self.fp_generator = fingerprint_generator_factory
        self.file_extensions = file_extensions
        self.num_processes = num_processes
        self.chunk_size = (chunk_size, fingerprint_generator_factory.atom_length)
        self.queue_size = queue_size

    @staticmethod
    def _process_lines(
        in_queue: Queue,
        out_queue: Queue,
        leaf_map: LeafMap,
        generator_factory: DataloaderFingerprintGeneratorFactory,
        ranges_dict: RangesDict,
        ranges_lock,
    ) -> None:
        """
        Process lines from the input queue, generate fingerprints, and put results in the output queue.

        Parameters
        ----------
        in_queue : Queue
        out_queue : Queue
        leaf_map : LeafMap
            Description of the input file structure, mapping extra columns from the file to the necessary type and name
        generator : DataLoaderMolGenerator
            Generator for the fingerprints, which provides the get_fingerprint method
        mins : list[float]
            Managed list to store the minimum fingerprint values from each worker
        maxs : list[float]
            Managed list to store the maximum fingerprint values from each worker
        """
        local_gen = generator_factory.get_generator()

        local_range_dict = ranges_dict.copy()

        # Remove the primary column from the leaf map
        # and get the mol_string_column
        mol_string_column = leaf_map.pop("primary")[0]

        while True:
            try:
                # FIXME: Can throw a broken pipe error on exit. Seems to cause no harm
                batch = in_queue.get()
            except ShutDown:
                break
            else:
                out_batch: dict = {
                    "smiles": [],
                    "fingerprint": np.empty(
                        shape=(0, generator_factory.fingerprint_length),
                        dtype=generator_factory.atom_dtype,
                    ),
                }

                for leaf_name, leaf_properties in leaf_map.items():
                    # If string type, initialize as empty list
                    # If other type, initialize as empty array
                    if leaf_properties[1] is str:
                        out_batch[leaf_name] = []
                    else:
                        out_batch[leaf_name] = np.empty((0,), dtype=leaf_properties[1])

                for element in batch:
                    try:
                        smiles, fingerprint = local_gen.get_fingerprint(
                            element, mol_string_column
                        )
                        out_batch["smiles"].append(smiles)
                        out_batch["fingerprint"] = np.vstack(
                            [
                                out_batch["fingerprint"],
                                fingerprint[np.newaxis, :],
                            ],
                        )
                    except Exception:
                        # On exception, skip the current element
                        # Maybe add an exception counter, to see how many elements failed
                        continue
                    else:
                        for leaf_name, leaf_properties in leaf_map.items():
                            dtype = leaf_properties[1]
                            property_position = leaf_properties[0]
                            value = element[property_position]
                            try:
                                # If string type, append to list
                                if dtype is str:
                                    out_batch[leaf_name].append(value)
                                # If other type, append to array
                                else:
                                    item = np.array(
                                        dtype(value),
                                        dtype=dtype,
                                    )
                                    out_batch[leaf_name] = np.concat(
                                        [
                                            out_batch[leaf_name],
                                            item[np.newaxis],
                                        ]
                                    )
                            except Exception:
                                print(
                                    f"Parsing of leaf {leaf_name}, with type {dtype} failed with value: {value}"
                                )
                                continue

                for leaf_name, values in out_batch.items():
                    local_range_dict_item = local_range_dict[leaf_name]
                    if local_range_dict_item["type"] == "continous":
                        local_range_dict_item["value_range"][0] = min(
                            local_range_dict_item["value_range"][0], np.min(values)
                        )
                        local_range_dict_item["value_range"][1] = max(
                            local_range_dict_item["value_range"][0], np.max(values)
                        )
                    elif local_range_dict_item["type"] == "categorical":
                        local_range_dict_item["value_range"] = set.union(
                            local_range_dict_item["value_range"], set(values)
                        )

                # Packing after creating the whole batch, so that we need to determine min and max only once per batch and not on every line
                out_batch["fingerprint"] = generator_factory.packer(
                    out_batch["fingerprint"]
                )
                out_queue.put(out_batch, block=True, timeout=None)
                in_queue.task_done()

        ranges_lock.acquire()
        for leaf_name, local_values in local_range_dict.items():
            if local_values["type"] == "continous":
                local_values["value_range"][0] = min(
                    local_values["value_range"][0],
                    ranges_dict[leaf_name]["value_range"][0],
                )
                local_values["value_range"][1] = max(
                    local_values["value_range"][1],
                    ranges_dict[leaf_name]["value_range"][1],
                )
            elif local_values["type"] == "categorical":
                local_values["value_range"] = set(
                    [str(value) for value in local_values["value_range"]]
                )  # convert all to string, as other dtype lead to problem with the gui
                local_values["value_range"] = set.union(
                    local_values["value_range"], ranges_dict[leaf_name]["value_range"]
                )
            ranges_dict[leaf_name] = local_values
        ranges_lock.release()

    @staticmethod
    @abstractmethod
    def _create_leaf_structure(
        root: FileRoot,
        group_names: list[str],
        fingerprint_chunk_size: tuple[int, int],
        fingerprint_dtype: type,
        leaf_map: LeafMap,
        ranges_dict: RangesDict,
    ) -> None:
        pass

    @staticmethod
    @abstractmethod
    def _write_lines(queue: Queue, progress_bar: tqdm, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def create(
        self,
        file_hierarchy: FileList,
        out_path: str,
        leaf_map: LeafMap,
        skip_lines: int = 0,
        sep: str = "\t",
    ) -> None:
        pass

    @staticmethod
    def _create_ranges_dict(leaf_map: LeafMap):
        _ = leaf_map.pop("primary")
        ranges_dict = {}
        ranges_dict["fingerprint"] = {
            "type": "continous",
            "value_range": np.array([np.inf, -np.inf]),
        }
        ranges_dict["smiles"] = {"type": "na", "value_range": []}

        for leaf_name, leaf_properties in leaf_map.items():
            try:
                gui_use = leaf_properties[2]
                if gui_use == "continous":
                    ranges_dict[leaf_name] = {
                        "type": "continous",
                        "value_range": np.array([np.inf, -np.inf]),
                    }

                elif gui_use == "categorical":
                    ranges_dict[leaf_name] = {
                        "type": "categorical",
                        # TODO: use of set() is suboptimal here as set it is not a native dtype. Probably should change to list or array to ensure compatablity
                        "value_range": set(),
                    }

                elif gui_use == "na":
                    ranges_dict[leaf_name] = {"type": "na", "value_range": []}

            except IndexError:
                ranges_dict[leaf_name] = {"type": "na", "value_range": []}

        return ranges_dict

    @staticmethod
    @abstractmethod
    def _write_ranges_dict(
        root: FileRoot,
        group_name: str,
        ranges_dict: RangesDict,
    ) -> None:
        pass


class HDF5Creator(StoreCreator):
    """
    HDF5 Store Create for creating HDF5 file of cheminformatic datasets from file hierarchies.
    """

    def __init__(
        self,
        fingerprint_generator_factory: DataloaderFingerprintGeneratorFactory,
        file_extensions: list[str] = [".txt", ".smi", ".csv"],
        num_processes: int = (mp.cpu_count() - 2),
        chunk_size: int = 1000,
        queue_size: int = 500,
    ):
        """
        Initialized the HDF5Creator

        Parameters
        ----------
        fingerprint_generator_factory : DataloaderFingerprintGeneratorFactory
            Factory to use, depends on the input file type and content.
        file_extensions : list[str], optional
            File extentions to consider.
        num_processes : int, optional
            Number of processes used.
        chunk_size : int, optional
            Size of chunks send to processes.
            Should not need configuration.
        queue_size : int, optional
            Size of queue of each process,.
            Should not need configuration.
        """
        super(HDF5Creator, self).__init__(
            fingerprint_generator_factory=fingerprint_generator_factory,
            file_extensions=file_extensions,
            num_processes=num_processes,
            chunk_size=chunk_size,
            queue_size=queue_size,
        )

    @staticmethod
    def _create_leaf_structure(
        root: tables.Group,
        group_names: list[str],
        fingerprint_chunk_size: tuple[int, int],
        fingerprint_dtype: type,
        leaf_map: LeafMap,
        ranges_dict: RangesDict,
    ) -> None:
        _ = leaf_map.pop("primary")

        for group in group_names:
            h5_group = tables.Group(root, group, new=True)
            fp_atom = tables.Atom.from_dtype(
                np.dtype((fingerprint_dtype, fingerprint_chunk_size[1]))
            )
            # Create arrays for fingerprints and smiles
            fp_array = tables.EArray(
                h5_group,
                "fingerprint",
                atom=fp_atom,
                shape=(0,),
                chunkshape=(fingerprint_chunk_size[0],),
            )
            ranges_info = ranges_dict["fingerprint"]
            fp_array.attrs["type"] = ranges_info["type"]
            fp_array.attrs["value_range"] = ranges_info["value_range"]

            smiles_array = tables.VLArray(
                h5_group,
                "smiles",
                atom=tables.VLUnicodeAtom(),
                chunkshape=(fingerprint_chunk_size[0],),
            )
            ranges_info = ranges_dict["smiles"]
            smiles_array.attrs["type"] = ranges_info["type"]
            smiles_array.attrs["value_range"] = ranges_info["value_range"]

            for leaf_name, leaf_properties in leaf_map.items():
                if leaf_properties[1] is not str:
                    leaf_atom = tables.Atom.from_dtype(np.dtype(leaf_properties[1]))
                    new_array = tables.EArray(
                        h5_group,
                        leaf_name,
                        atom=leaf_atom,
                        shape=(0,),
                        chunkshape=(fingerprint_chunk_size[0],),
                    )
                else:  # If the type is string, create a VLArray
                    new_array = tables.VLArray(
                        h5_group,
                        leaf_name,
                        atom=tables.VLUnicodeAtom(),
                        chunkshape=(fingerprint_chunk_size[0],),
                    )

                new_array.attrs["type"] = ranges_dict[leaf_name]["type"]
                new_array.attrs["value_range"] = ranges_dict[leaf_name]["value_range"]

    @staticmethod
    def _write_ranges_dict(
        root: FileRoot,
        group_name: str,
        ranges_dict: RangesDict,
    ) -> None:
        with tables.open_file(root, mode="a") as h5_file:
            group = h5_file.root[group_name]
            for name, ranges in ranges_dict.items():
                array = group[name]
                array.attrs["type"] = ranges["type"]
                array.attrs["value_range"] = ranges["value_range"]

    @staticmethod
    def _write_lines(
        queue: Queue,
        progress_bar: tqdm,
        h5_filename: str,
        group_name: str,
    ) -> None:
        with tables.open_file(h5_filename, mode="a") as h5_file:
            h5_group = h5_file.root[group_name]
            while True:
                try:
                    new_item = queue.get()
                except ShutDown:
                    break
                else:
                    for leaf_name, content in new_item.items():
                        if type(content[0]) is not str:
                            h5_group[leaf_name].append(content)
                        else:
                            for item in content:
                                h5_group[leaf_name].append(item)

                    progress_bar.update(len(content))
                    queue.task_done()

    def create(
        self,
        file_hierarchy: FileList,
        out_path: str,
        leaf_map: LeafMap,
        sep: str,
        skip_lines: int = 0,
    ) -> None:
        """
        Run creation of HDF5 storage file

        Parameters
        ----------
        file_hierarchy :
            Dictionary of files to parse, see How-To Guides.
        out_path :
            Output path for the HDF5 files.
        leaf_map :
            Dictionary of file structure and datatypes, see How-To Guides.
        sep :
            Column seperator.
        skip_lines :
            Number of lines to skip at the beginning of file, e.g. for headers.
        """
        file_hierarchy = _parse_file_hierarchy(file_hierarchy, self.file_extensions)
        out_path = _parse_output_path(out_path, [".h5", ".hdf5"])
        ranges_dict_prototype = self._create_ranges_dict(leaf_map.copy())

        with tables.open_file(out_path, mode="w", title="chisom HDF5 Store") as h5_file:
            h5_root = h5_file.root

            h5_root._v_attrs["fingerprint_length"] = (
                self.fp_generator.fingerprint_length
            )
            h5_root._v_attrs["packed_flag"] = self.fp_generator.packed

            self._create_leaf_structure(
                h5_root,
                list(file_hierarchy.keys()),
                self.chunk_size,
                self.fp_generator.atom_dtype,
                leaf_map.copy(),
                ranges_dict_prototype.copy(),
            )

        for group_name, files in file_hierarchy.items():
            filegroup = fileinput.input(files)
            print("Processing group:", group_name)
            num_lines = estimate_lines(files)
            print(f"Estimated number of lines: {num_lines}")
            batches = LazyIterator(
                filegroup,
                num_lines,
                delimiter=sep,
                batch_size=self.chunk_size[0],
                skip_lines=skip_lines,
            )
            with mp.Manager() as manager:
                process_queue = manager.Queue(maxsize=self.queue_size)
                write_queue = manager.Queue(maxsize=self.queue_size)
                ranges_dict = manager.dict(ranges_dict_prototype.copy())
                ranges_lock = manager.Lock()

                progress_bar = tqdm(total=num_lines, desc="Processing", unit="lines")

                writing_process = mp.Process(
                    target=self._write_lines,
                    args=(write_queue, progress_bar, out_path, group_name),
                )
                writing_process.start()

                mol_processes = []
                for _ in range(self.num_processes):
                    new_proc = mp.Process(
                        target=self._process_lines,
                        args=(
                            process_queue,
                            write_queue,
                            leaf_map.copy(),
                            self.fp_generator,
                            ranges_dict,
                            ranges_lock,
                        ),
                    )
                    mol_processes.append(new_proc)
                    new_proc.start()

                for batch in batches:
                    try:
                        process_queue.put(batch)
                    except Exception as e:
                        process_queue.shutdown(immediate=True)
                        write_queue.shutdown(immediate=True)
                        raise e
                process_queue.shutdown()
                process_queue.join()
                write_queue.shutdown()
                write_queue.join()
                writing_process.join()

                self._write_ranges_dict(out_path, group_name, ranges_dict)

            filegroup.close()
