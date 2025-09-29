"""pyHIPIFY package."""

from .hipify_python import hipify, process, preprocessor
from .cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS

# Import the short names module
from . import short

__all__ = [
    'hipify',
    'process', 
    'preprocessor',
    'CUDA_TO_HIP_MAPPINGS',
    'short'
]

__version__ = '0.0.1a2'

"""pyHIPIFY: A tool for converting CUDA source code to HIP source code.

This module provides tools for automated conversion of CUDA code to HIP code,
which allows code originally written for NVIDIA GPUs to run on AMD GPUs.

Basic usage:
------------

For simple file conversion:
>>> from pyHIPIFY import hipify
>>> hipify.convert("/path/to/cuda/project", output_directory="/path/to/hip/project")

For single file processing:
>>> from pyHIPIFY import api
>>> api.transform("input.cu", "output.hip")

Main components:
---------------

1. Constants: Predefined mappings and constants used in the conversion process
2. Functions: Main functions for converting CUDA code to HIP code
3. Classes: Supporting classes for the conversion process
4. Exceptions: Custom exceptions for error handling
5. Submodules: Additional functionality organized in submodules

"""

# Version info
__version__ = '0.0.1a8'

# Import main functions with shorter names
from . import api
from .hipify_python import hipify, process, preprocessor

# For backward compatibility
convert_project = hipify
convert_file = process

# Import constants and mappings
from .cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS
from .constants import *

# Define what's available in the public API
__all__ = [
    'api',
    'hipify',
    'process',
    'preprocessor',
    'convert_project',
    'convert_file',
    'CUDA_TO_HIP_MAPPINGS'
]