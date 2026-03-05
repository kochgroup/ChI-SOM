from numpy import float32
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator

from chisom.io.datastore_creation import HDF5Creator
from chisom.io.datastore_factories import rdStyleFactory

generator = rdFingerprintGenerator.GetMorganGenerator
fingerprint_kwargs = {"fpSize": 1024, "radius": 2}

file_dict = {
    "active": [
        "tests/testdata/VDR/actives.smi",
    ],
    "inactive": [
        "tests/testdata/VDR/inactives.smi",
    ],
}

molgen = rdStyleFactory(
    MolFromSmiles,
    generator,
    generator_kwargs=fingerprint_kwargs,
    count_fingerprint=True,
)
file_creator = HDF5Creator(fingerprint_generator_factory=molgen)

leaf_map = {
    "primary": (0, str),
    "ID": (1, str, "na"),
    "Activity": (2, int, "categorical"),
    "MolWt": (3, float32, "continous"),
    "MolLogP": (4, float32, "continous"),
    "TPSA": (5, float32, "continous"),
}

file_creator.create(
    file_dict,
    "tests/testdata/VDR.h5",
    leaf_map,
    skip_lines=1,
    sep="\t",
)
