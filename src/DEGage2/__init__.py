"""DEGage2: Identification of differentially expressed genes using DOTNB."""

# Import key entry point functions from modules inside the package
from .run_DEGage2 import DEGage2_0, DOTNB
from .preprocessing import preprocess, permutation
from .parameters import calculate_r, calculate_k_bootstrapped
from .DOTNB import DOTNB_pmf, DOTNB_cdf

__version__ = "0.1.0"

# Explicitly define what gets exported when using 'from degage2 import *'
__all__ = [
    "DEGage2_0",
    "DOTNB",
    "preprocess",
    "permutation",
    "calculate_r",
    "calculate_k_bootstrapped",
    "DOTNB_pmf",
    "DOTNB_cdf",
]