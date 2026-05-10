"""Basic objects that can cache rasterized / derived data."""

import copy
import logging
import os
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn

import dask
import dask.array as da
import h5py
import pandas as pd
from rex.utilities.utilities import to_records_array

from sup3r.preprocessing.base import Container
from sup3r.preprocessing.names import Dimension
from sup3r.preprocessing.utilities import _lowered, _mem_check
from sup3r.utilities.utilities import (
    encode_str_times,
    get_tmp_file,
    safe_cast,
    safe_serialize,
)

from .utilities import _check_for_cache

if TYPE_CHECKING:
    from sup3r.preprocessing.accessor import Sup3rX
    from sup3r.preprocessing.base import Sup3rDataset


logger = logging.getLogger(__name__)


class Cacher(Container):
    """Base cacher object. Simply writes given data to H5 or NETCDF files. By
    default every feature will be written to a separate file. To write multiple
    features to the same file call :meth:`write_netcdf` or :meth:`write_h5`
    directly"""

    def __init__(
        self,
        data: Union['Sup3rX', 'Sup3rDataset'],
        cache_kwargs: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        data : Union[Sup3rX, Sup3rDataset]
            Data to write to file
        cache_kwargs : dict
            Dictionary with kwargs for caching wrangled data. This should at
            minimum include a 'cache_pattern' key, value. This pattern must
            have a {feature} format key and either a h5 or nc file extension,
            based on desired output type.

            Can also include a ``max_workers``, ``chunks``, and ``attrs`` keys.
            ``max_workers`` is an inteeger specifying number of threads to use
            for writing chunks to output files, ``chunks`` is a dictionary
            of dictionaries for each feature (or a single dictionary to use
            for all features), and ``attrs`` is a dictionary of attributes to
            add to the output files which can include ``scale_factor`` values.
            e.g.
            .. code-block:: JSON
                {'cache_pattern': ...,
                    'chunks': {
                        'u_10m': {
                            'time': 20,
                            'south_north': 100,
                            'west_east': 100
                        }
                    }
                    'attrs': {
                        'u_10m': {'scale_factor': 10}
                    }
                }

        Note
        ----
        This is only for saving cached data. If you want to reload the
        cached files load them with a ``Loader`` object. ``DataHandler``
        objects can cache and reload from cache automatically.
        """
        super().__init__(data=data)
        if (
            cache_kwargs is not None
            and cache_kwargs.get('cache_pattern', None) is not None
        ):
            self.out_files = self.cache_data(**cache_kwargs)

    @classmethod
    def _write_single(
        cls,
        out_file,
        data,
        features='all',
        chunks=None,
        max_workers=1,
        mode='w',
        attrs=None,
        overwrite=False,
        time_last=None,
    ):
        """Write single NETCDF or H5 cache file."""
        if os.path.exists(out_file) and not overwrite:
            logger.info(
                f'{out_file} already exists. Delete or specify overwrite=True '
                'if you want to overwrite.'
            )
            return
        if features == 'all':
            features = list(data.data_vars)
        features = features if isinstance(features, list) else [features]
        _, ext = os.path.splitext(out_file)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        tmp_file = get_tmp_file(out_file)
        logger.debug(
            'Writing %s to %s with max_workers=%s. %s',
            features,
            tmp_file,
            max_workers,
            _mem_check(),
        )
        if ext == '.h5':
            func = cls.write_h5
            time_last = time_last if time_last is not None else False
        elif ext == '.nc':
            func = cls.write_netcdf
            time_last = time_last if time_last is not None else True
        else:
            msg = (
                'cache_pattern must have either h5 or nc extension. '
                f'Received {ext}.'
            )
            logger.error(msg)
            raise ValueError(msg)
        func(
            out_file=tmp_file,
            data=data,
            features=features,
            chunks=chunks,
            max_workers=max_workers,
            mode=mode,
            attrs=attrs,
            time_last=time_last,
        )
        os.replace(tmp_file, out_file)
        logger.debug('Moved %s to %s', tmp_file, out_file)

    def cache_data(
        self,
        cache_pattern,
        chunks=None,
        max_workers=None,
        mode='w',
        attrs=None,
        time_last=None,
        overwrite=False,
    ):
        """Cache data to file with file type based on user provided
        cache_pattern.

        Parameters
        ----------
        cache_pattern : str
            Cache file pattern. Must have a {feature} format key. The extension
            (.h5 or .nc) specifies which format to use for caching.
        chunks : dict | None
            Chunk sizes for coordinate dimensions. e.g.
            ``{'u_10m': {'time': 10, 'south_north': 100, 'west_east': 100}}``
        max_workers : int | None
            Number of workers to use for parallel writing of chunks
        mode : str
            Write mode for ``out_file``. Defaults to write.
        attrs : dict | None
            Optional attributes to write to file. Can specify dataset specific
            attributes by adding a dictionary with the dataset name as a key.
            e.g. {**global_attrs, dset: {...}}
        time_last : bool
            Whether to keep the original dimension order of the data. If
            ``False`` then the data will be transposed to have the time
            dimension first. This defaults to False for H5 output and True for
            NetCDF output.
        overwrite : bool
            Whether to overwrite existing cache files.
        """
        msg = 'cache_pattern must have {feature} format key.'
        assert '{feature}' in cache_pattern, msg

        cached_files, _, missing_files, missing_features = _check_for_cache(
            features=self.features,
            cache_kwargs={'cache_pattern': cache_pattern},
        )

        if any(cached_files) and not overwrite:
            logger.info(
                f'Cache files with pattern {cache_pattern} already exist. '
                'Delete or specify overwrite=True to overwrite.'
            )
        elif any(cached_files) and overwrite:
            missing_files += cached_files

        if any(missing_files):
            logger.info('Caching %s to %s', missing_features, missing_files)
            for feature, out_file in zip(missing_features, missing_files):
                self._write_single(
                    data=self.data,
                    features=feature,
                    out_file=out_file,
                    chunks=chunks,
                    max_workers=max_workers,
                    mode=mode,
                    attrs=attrs,
                    time_last=time_last,
                    overwrite=overwrite,
                )
            logger.debug('Finished writing %s', missing_files)
        return missing_files + cached_files

    @staticmethod
    def parse_chunks(feature, chunks, dims):
        """Parse chunks input to Cacher. Needs to be a dictionary of dimensions
        and chunk values but parsed to a tuple for H5 caching."""

        if isinstance(chunks, dict) and feature.lower() in _lowered(chunks):
            chunks = {k.lower(): v for k, v in chunks.items()}
            fchunks = chunks.get(feature, {})
        else:
            fchunks = copy.deepcopy(chunks)
        if isinstance(fchunks, dict):
            fchunks = {d: fchunks.get(d, None) for d in dims}
        if isinstance(fchunks, int):
            fchunks = {feature: fchunks}
        if any(chk is None for chk in fchunks):
            fchunks = 'auto'
        return fchunks

    @classmethod
    def get_chunksizes(cls, dset, data, chunks):
        """Get chunksizes after rechunking (could be undetermined beforehand
        if ``chunks == 'auto'``) and return rechunked data.

        Parameters
        ----------
        dset : str
            Name of feature to get chunksizes for.
        data : Sup3rX | xr.Dataset
            ``Sup3rX`` or ``xr.Dataset`` containing data to be cached.
        chunks : dict | None | 'auto'
            Dictionary of chunksizes either to use for all features or, if the
            dictionary includes feature keys, feature specific chunksizes. Can
            also be None or 'auto'.
        """
        data_var = data.coords[dset] if dset in data.coords else data[dset]
        fchunk = cls.parse_chunks(dset, chunks, data_var.dims)
        if isinstance(fchunk, dict):
            fchunk = {k: v for k, v in fchunk.items() if k in data_var.dims}

        data_var = data_var.chunk(fchunk)
        data_var = data_var.unify_chunks()

        chunksizes = tuple(d[0] for d in data_var.chunksizes.values())
        chunksizes = chunksizes if chunksizes else None

        return data_var, chunksizes

    @classmethod
    def _check_dim_order(cls, data, time_last):
        """Check dim order and put time first if `time_last` is False."""

        # Transpose time to first dimension for common hdf convention
        if not time_last and Dimension.TIME in data.dims:
            if len(data.dims) == 3:
                data = data.transpose(Dimension.TIME, *Dimension.dims_2d())
            elif len(data.dims) == 2:
                data = data.transpose(
                    Dimension.TIME, Dimension.FLATTENED_SPATIAL
                )
        return data

    @classmethod
    def _prepare_h5_data(cls, data, features):
        """Prepare dataset and feature list for H5 writing."""

        if features == 'all':
            features = list(data.data_vars)
        features = features if isinstance(features, list) else [features]

        data_subset = data[features].copy()

        if Dimension.TIME in data_subset.coords:
            times = encode_str_times(data_subset[Dimension.TIME].values)
            data_subset = data_subset.assign({Dimension.TIME: times})

        return data_subset, features

    @classmethod
    def _set_h5_meta(cls, f, data_subset, attrs):
        """Get global attrs and write meta dataset to the open H5 file."""
        attrs = attrs or {}
        meta = attrs.pop('meta', {})

        if meta is None or (isinstance(meta, dict) and not meta):
            meta_df = pd.DataFrame()
        elif isinstance(meta, pd.DataFrame):
            meta_df = meta.copy()
        else:
            meta_df = pd.DataFrame(meta)
        for coord in Dimension.coords_2d():
            if coord in data_subset:
                meta_df[coord] = data_subset[coord].data.flatten()

        meta_rec = to_records_array(meta_df)
        f.create_dataset(
            '/meta',
            shape=meta_rec.shape,
            dtype=meta_rec.dtype,
            data=meta_rec,
        )

    @classmethod
    def _get_h5_array_and_chunks(cls, dset, data, chunks):
        """Get dask array for dataset and chunksizes for H5 writing."""
        data_var, chunksizes = cls.get_chunksizes(dset, data, chunks)
        data_arr = data_var.data

        if not isinstance(data_arr, da.core.Array):
            data_arr = da.asarray(data_arr)

        dset_name = dset
        if dset == Dimension.TIME:
            dset_name = 'time_index'

        return data_arr, chunksizes, dset_name

    @classmethod
    def _scale_data(cls, data_subset, out_file):
        """Scale data array based on scale_factor attribute."""
        for dset in data_subset.data_vars:
            scale_factor = data_subset[dset].attrs.get('scale_factor', None)
            if scale_factor is not None and scale_factor != 1:
                _, ext = os.path.splitext(out_file.replace('.tmp', ''))
                if ext == '.h5':
                    data_subset[dset] *= scale_factor
                else:
                    data_subset[dset] /= scale_factor
        return data_subset

    @classmethod
    def write_h5(
        cls,
        out_file,
        data,
        features='all',
        chunks=None,
        max_workers=None,
        mode='w',
        attrs=None,
        time_last=False,
        write_coords=True,
    ):
        """Cache data to h5 file using user provided chunks value.

        Parameters
        ----------
        out_file : str
            Name of file to write. Must have a .h5 extension.
        data : Sup3rDataset | Sup3rX | xr.Dataset
            Data to write to file. Comes from ``self.data``, so an
            ``xr.Dataset`` like object with ``.dims`` and ``.coords``
        features : str | list
            Name of feature(s) to write to file.
        chunks : dict | None
            Chunk sizes for coordinate dimensions. e.g.
            ``{'u_10m': {'time': 10, 'south_north': 100, 'west_east': 100}}``
        max_workers : int | None
            Number of workers to use for parallel writing of chunks
        mode : str
            Write mode for ``out_file``. Defaults to write.
        attrs : dict | None
            Optional attributes to write to file. Can specify dataset specific
            attributes by adding a dictionary with the dataset name as a key.
            e.g. {**global_attrs, dset: {...}}. Can also include a global meta
            dataframe that will then be added to the coordinate meta.
        time_last : bool
            Whether to keep time as the last dimension. If ``False`` then the
            data will be transposed to have the time dimension first.
        write_coords : bool
            Whether to write coordinate datasets to file. If this is False
            only the requested features will be written. Specific coordinates
            can be written by including them in the features list.

        """
        chunks = chunks or 'auto'

        data_subset = cls._check_dim_order(data, time_last)

        data_subset, features = cls._prepare_h5_data(
            data=data_subset, features=features
        )

        coord_names = (
            [crd for crd in data_subset.coords if crd in Dimension.coords_4d()]
            if write_coords
            else []
        )

        with h5py.File(out_file, mode) as f:
            data_subset, global_attrs = cls._set_data_attributes(
                data_subset, features, attrs
            )
            data_subset = cls._scale_data(data_subset, out_file=out_file)
            cls._set_h5_meta(f, data_subset, attrs)
            for k, v in global_attrs.items():
                f.attrs[k] = safe_cast(v)

            for dset in [*coord_names, *features]:
                data_arr, chunksizes, dset_name = cls._get_h5_array_and_chunks(
                    dset, data_subset, chunks
                )

                dset_obj = f.create_dataset(
                    f'/{dset_name}',
                    dtype=data_arr.dtype,
                    shape=data_arr.shape,
                    chunks=chunksizes,
                )
                for k, v in data_subset[dset].attrs.items():
                    dset_obj.attrs[k] = safe_cast(v)

                if max_workers == 1:
                    da.store(data_arr, dset_obj, scheduler='single-threaded')
                else:
                    da.store(
                        data_arr,
                        dset_obj,
                        scheduler='threads',
                        num_workers=max_workers,
                    )

    @classmethod
    def _prepare_netcdf_data(cls, data, features, chunks):
        """Prepare data subset for netCDF writing."""
        if features == 'all':
            features = list(data.data_vars)
        features = features if isinstance(features, list) else [features]

        data_subset = data[features].copy()

        coord_names = [
            crd for crd in data.coords if crd in Dimension.coords_4d()
        ]
        for coord_name in coord_names:
            if coord_name not in data_subset.coords:
                data_subset = data_subset.assign_coords({
                    coord_name: data[coord_name]
                })

        if chunks != 'auto':
            for feature in features:
                fchunk = cls.parse_chunks(
                    feature, chunks, data_subset[feature].dims
                )
                if isinstance(fchunk, dict):
                    fchunk = {
                        k: v
                        for k, v in fchunk.items()
                        if k in data_subset[feature].dims
                    }
                data_subset[feature] = data_subset[feature].chunk(fchunk)
        else:
            data_subset = data_subset.chunk('auto')

        return data_subset, features

    @classmethod
    def _set_data_attributes(cls, data_subset, features, attrs):
        """Set variable and global attributes for data output."""

        attrs = attrs or {}
        for feature in features:
            var_attrs = data_subset[feature].attrs.copy()
            var_attrs.update(attrs.pop(feature, {}))
            data_subset[feature].attrs.update(var_attrs)

        global_attrs = data_subset.attrs.copy()
        global_attrs.update(attrs)

        for attr_name, attr_value in global_attrs.items():
            attr_value = safe_cast(attr_value)
            try:
                data_subset.attrs[attr_name] = attr_value
            except Exception as e:
                msg = (
                    f'Could not write {attr_name} as attribute, '
                    f'serializing with json dumps, '
                    f'received error: "{e}"'
                )
                logger.warning(msg)
                warn(msg)
                data_subset.attrs[attr_name] = safe_serialize(attr_value)

        return data_subset, global_attrs

    @classmethod
    def _configure_netcdf_encoding(cls, data_subset, features, chunks):
        """Configure encoding for netCDF output."""
        encoding = {}
        for feature in features:
            _, chunksizes = cls.get_chunksizes(feature, data_subset, chunks)
            if chunksizes is not None:
                encoding[feature] = {
                    'chunksizes': chunksizes,
                }
        return encoding

    @classmethod
    def write_netcdf(
        cls,
        out_file,
        data,
        features='all',
        chunks=None,
        max_workers=1,
        mode='w',
        attrs=None,
        time_last=True,
    ):
        """Cache data to a netcdf file using xarray.

        Parameters
        ----------
        out_file : str
            Name of file to write. Must have a ``.nc`` extension.
        data : Sup3rDataset
            Data to write to file. Comes from ``self.data``, so a
            ``Sup3rDataset`` with coords attributes
        features : str | list
            Names of feature(s) to write to file.
        chunks : dict | None
            Chunk sizes for coordinate dimensions. e.g. ``{'south_north': 100,
            'west_east': 100, 'time': 10}`` Can also include dataset specific
            values. e.g. ``{'windspeed': {'south_north': 100, 'west_east': 100,
            'time': 10}}``
        max_workers : int | None
            Number of workers to use for parallel writing of chunks
        mode : str
            Write mode for ``out_file``. Defaults to write.
        attrs : dict | None
            Optional attributes to write to file. Can specify dataset specific
            attributes by adding a dictionary with the dataset name as a key.
            e.g. {**global_attrs, dset: {...}}
        time_last : bool
            Whether to keep time as the last dimension. If ``False`` then the
            data will be transposed to have the time dimension first.
        """
        chunks = chunks or 'auto'

        data_subset = cls._check_dim_order(data, time_last)

        data_subset, features = cls._prepare_netcdf_data(
            data_subset, features, chunks
        )

        data_subset, _ = cls._set_data_attributes(data_subset, features, attrs)

        data_subset = cls._scale_data(data_subset, out_file=out_file)

        encoding = cls._configure_netcdf_encoding(
            data_subset, features, chunks
        )

        if max_workers == 1:
            data_subset.to_netcdf(
                out_file,
                mode=mode,
                format='NETCDF4',
                encoding=encoding,
                compute=True,
            )

        else:
            with dask.config.set(scheduler='threads', num_workers=max_workers):
                data_subset.to_netcdf(
                    out_file,
                    mode=mode,
                    format='NETCDF4',
                    encoding=encoding,
                    compute=True,
                )

        logger.debug('Finished writing %s to %s', features, out_file)
