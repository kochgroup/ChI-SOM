from importlib.metadata import version

from ._som import Som
from ._interface.gui import start_chisom_viewer

__all__ = [
    "Som",
    "start_chisom_viewer",
]

__version__ = version("chi-som")
__author__ = "Johannes Kaminski"
__credits__ = "AG Koch"
