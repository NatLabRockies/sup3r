"""Container subclass with additional methods for loading the contained
data."""

from typing import ClassVar

from sup3r.preprocessing.base import Sup3rMeta
from sup3r.preprocessing.utilities import get_source_type

from .base import BaseLoader
from .h5 import LoaderH5
from .xr import LoaderX


class Loader(BaseLoader, metaclass=Sup3rMeta):
    """`Loader` class which parses input file type and returns
    appropriate `TypeSpecificLoader`."""

    TypeSpecificClasses: ClassVar = {
        'nc': LoaderX,
        'zarr': LoaderX,
        'h5': LoaderH5,
    }

    def __new__(cls, file_paths, **kwargs):
        """Override parent class to return type specific class based on
        `source_files`"""
        fp_type = get_source_type(file_paths)
        assert fp_type in cls.TypeSpecificClasses, (
            f'Unsupported file_paths: {file_paths}'
        )
        SpecificClass = cls.TypeSpecificClasses[fp_type]
        return SpecificClass(file_paths, **kwargs)

    _signature_objs = (BaseLoader,)
    __doc__ = BaseLoader.__doc__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self._res.close()
