from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from numpy.typing import NDArray
from rdkit.Chem import MolToSmiles

from chisom.io._types import (
    FingerprintStack,
    InputLine,
    MolGenerator,
    Packer,
    Smiles,
    rdFingerprintGenerator,
)


class UnifiedGenerator(ABC):
    """
    Abstract base class for unified fingerprint generators. This class defines the interface for generating fingerprints and molecular structures as smiles from a row of input Data.
    It is intended to be subclassed by specific fingerprint generator implementations.
    """

    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def get_fingerprint(
        self, row: InputLine, mol_string_column: int
    ) -> tuple[Smiles, NDArray]: ...


class DataloaderFingerprintGeneratorFactory(ABC):
    """
    Abstract base class for fingerprint generators factories. Should supply individual fingerprint generators for use in DataLoader creation.
    Is intended to be subclassed by specific fingerprint generator implementations.
    """

    packer: Packer

    def __init__(self, atom_dtype: type):
        self.packer = self._generate_packer_stub(atom_dtype)

    @property
    @abstractmethod
    def fingerprint_length(self) -> int: ...

    @property
    @abstractmethod
    def atom_length(self) -> int: ...

    @property
    @abstractmethod
    def atom_dtype(self) -> type: ...

    @property
    @abstractmethod
    def packed(self) -> bool: ...

    @abstractmethod
    def get_generator(self) -> UnifiedGenerator:
        """
        Returns a new instance of the fingerprint generator.
        This method should be implemented by subclasses to return a generator that can be used in a multiprocessing context.
        """
        pass

    @staticmethod
    def _generate_packer_stub(dtype: type) -> Packer:
        """
        Generates a stub for the packer function that simply converts the fingerprints to the specified dtype.
        This is used when the fingerprints are not binary and cannot be packed using np.packbits.
        """

        def _pack_stub(fp: FingerprintStack) -> NDArray:
            return np.asarray(fp, dtype=dtype)

        return _pack_stub

    def _is_packable(self, fingerprints: list[NDArray]) -> bool:
        """
        Checks if the fingerprints are binary (0 or 1) and can be packed using np.packbits.

        Parameters
        ----------
        fingerprints : list[NDArray]
            List of fingerprints to check.

        Returns
        -------
        bool
            True if the fingerprints are binary and can be packed, False otherwise.
        """
        is_binary = np.asarray([(fp == 0) | (fp == 1) for fp in fingerprints])
        return bool(is_binary.all())


class rdStyleFactory(DataloaderFingerprintGeneratorFactory):
    """
    This class is designed to provide a unified interface for generating molecular fingerprints for use in  DataLoader creation. It can combine different types of
    input (SMILES, InChI, ...) and fingerprint generator (e.g., Morgan, RDKit, Feature Morgan ... ).
    Also returns a new instance of the real generator on each call to get_generator, so that the generator can be used in a multiprocessing context without issues.
    """

    test_rows = [
        ["c1ccccc1"],
        ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
        ["CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"],
    ]

    def __init__(
        self,
        mol_generator: MolGenerator,
        fingerprint_generator: rdFingerprintGenerator,
        generator_kwargs={},
        count_fingerprint=False,
    ):
        """
        Initialize the Factory

        Parameters
        ----------
        mol_generator : MolGenerator
            The RDkit MolGenerator to use to read Molecules in the input file.
            e.g. MolFromSmiles
        fingerprint_generator : rdFingerprintGenerator
            The rdFingerprintGenerator to use.
        generator_kwargs : dict
            Keyword argument to pass to the rdFingerprintGenerator.
        count_fingerprint : bool
            If count fingerprints should be created where applicable.
        """

        self.generator_kwargs: dict = generator_kwargs

        self.generator = self._generator_factory(
            mol_generator, fingerprint_generator, count_fingerprint
        )

        try:
            test_gen = self.generator(self.generator_kwargs)
            test_gen_results: list[tuple[Smiles, NDArray]] = [
                test_gen.get_fingerprint(row, 0) for row in self.test_rows
            ]
            test_fps = [fp for _, fp in test_gen_results]
            self._fingerprint_length = test_fps[0].shape[0]
            self._packed = self._is_packable(test_fps)
            # Prepare either np.packbits or a stub for packing
            if self._packed:
                self.packer = partial(np.packbits, axis=-1)
                packed_fp = self.packer(np.asarray(test_fps))
            else:
                self.packer = self._generate_packer_stub(np.uint8)
                packed_fp = self.packer(test_fps)
            self._atom_length = packed_fp.shape[1]
            self._atom_dtype = np.uint8

            del packed_fp
            del test_gen
            del test_gen_results

        except Exception as e:
            print(f"Failed creating generator with:\n{str(e)}")
            exit(5)

    @property
    def fingerprint_length(self) -> int:
        return self._fingerprint_length

    @property
    def atom_length(self) -> int:
        return self._atom_length

    @property
    def atom_dtype(self) -> type:
        return self._atom_dtype

    @property
    def packed(self) -> bool:
        return self._packed

    def get_generator(self) -> UnifiedGenerator:
        # Returns a new instance of the fingerprint generator.
        return self.generator(self.generator_kwargs)

    @staticmethod
    def _generator_factory(
        mol_generator: MolGenerator,
        fp_generator: rdFingerprintGenerator,
        count: bool,
    ) -> type[UnifiedGenerator]:
        """
        Factory function to create a generator class to handle either rdkit style creation of fingerprints from
        a Mol object or CSV style creation of fingerprints from a row of data under the same interace.
        """

        # Set up generators init function, initializing the underlying generator with the kwargs
        def init(self, generator_kwargs):
            self.generator = fp_generator(**generator_kwargs)

        if count:

            def get_fp(
                self, row: InputLine, mol_string_column: int
            ) -> tuple[str, NDArray]:
                mol = mol_generator(row[mol_string_column])
                fingerprint = self.generator.GetCountFingerprintAsNumPy(mol)
                smile = MolToSmiles(mol)
                return smile, fingerprint
        else:

            def get_fp(
                self, row: InputLine, mol_string_column: int
            ) -> tuple[str, NDArray]:
                mol = mol_generator(row[mol_string_column])
                fingerprint = self.generator.GetFingerprintAsNumPy(mol)
                smile = MolToSmiles(mol)
                return smile, fingerprint

        # Create a new class that inherits from the UnifiedGenerator class
        # and set the init and get_fingerprint functions
        generator = type(
            "RDKitUnifiedGenerator",
            (UnifiedGenerator,),
            {"__init__": init, "get_fingerprint": get_fp},
        )
        return generator


class CSVStyleFactory(DataloaderFingerprintGeneratorFactory):
    """
    A generator that creates fingerprints from a row of data, not from a Mol object.
    With a similar interface to the RDKit fingerprint generator, but using a slice of the row instead of a Mol object.
    """

    def __init__(
        self,
        mol_generator: MolGenerator,
        fpStart: int,
        fpSize: int,
        dtype: type = np.float32,
    ):
        """
        Initializes the factory

        Parameters
        ----------
        mol_generator : MolGenerator
            The RDKit MolGenerator to use to read Molecules in the input file
            e.g. MolFromSmiles
        fpStart : int
            Numerical column index where the fingerprints starts.
        fpSize : int
            Length in number of columns of the CSV file of the fingerprint.
        dtype : type, optional
            Datatype of the fingperint elements
        """
        self.generator_kwargs = {"fpStart": fpStart, "fpSize": fpSize, "dtype": dtype}
        self._atom_dtype = dtype
        self._atom_length = fpSize
        self._fingerprint_length = fpSize
        self.mol_generator = mol_generator

        self.generator = self._generator_factory(mol_generator)
        super().__init__(self._atom_dtype)

    def get_generator(self) -> UnifiedGenerator:
        # Returns a new instance of the fingerprint generator.
        return self.generator(**self.generator_kwargs)

    @staticmethod
    def _generator_factory(mol_generator) -> type[UnifiedGenerator]:
        def init(self, fpStart, fpSize, dtype):
            self.start = fpStart
            self.stop = fpStart + fpSize
            self.dtype = dtype
            self.mol_generator = mol_generator

        def get_fp(self, row: InputLine, mol_string_column: int) -> tuple[str, NDArray]:
            mol = self.mol_generator(row[mol_string_column])
            smiles = MolToSmiles(mol)
            return smiles, np.asarray(row[self.start : self.stop], dtype=self.dtype)

        generator = type(
            "CSVUnifiedGenerator",
            (UnifiedGenerator,),
            {"__init__": init, "get_fingerprint": get_fp},
        )
        return generator

    @property
    def fingerprint_length(self) -> int:
        return self._fingerprint_length

    @property
    def atom_length(self) -> int:
        return self._atom_length

    @property
    def atom_dtype(self) -> type:
        return self._atom_dtype

    @property
    def packed(self) -> bool:
        return False
