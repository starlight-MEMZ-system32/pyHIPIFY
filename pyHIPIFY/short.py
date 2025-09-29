"""A simplified API with shorter names for common functions."""

from .hipify_python import hipify as convert_project
from .hipify_python import process as convert_file
from .hipify_python import preprocessor as preprocess

# Even shorter aliases
convert = convert_project
transform = convert_file
prep = preprocess

__all__ = [
    'convert_project',
    'convert_file', 
    'preprocess',
    'convert',
    'transform',
    'prep'
]