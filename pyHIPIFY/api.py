"""Simplified API for pyHIPIFY.

This module provides a simplified interface for the most common use cases
of the pyHIPIFY library with shorter function names.
"""

from .hipify_python import hipify as convert_project
from .hipify_python import process as convert_file
from .hipify_python import preprocessor as preprocess

# Shorter aliases for common functions
convert = convert_project
transform = convert_file
preprocess_file = preprocess

__all__ = [
    'convert',
    'convert_project',
    'convert_file',
    'transform',
    'preprocess',
    'preprocess_file'
]