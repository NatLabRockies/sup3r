"""NETCDF file collection.

TODO: Integrate this with Cacher class
"""

import logging
import os

import xarray as xr
from rex.utilities.loggers import init_logger

from sup3r.preprocessing.loaders import Loader
from sup3r.preprocessing.names import Dimension
from sup3r.writers import Cacher

from .base import BaseCollector

logger = logging.getLogger(__name__)


class CollectorNC(BaseCollector):
    """Sup3r NETCDF file collection framework"""

    @classmethod
    def collect(
        cls,
        file_paths,
        out_file,
        features='all',
        log_level=None,
        log_file=None,
        overwrite=True,
        res_kwargs=None,
        cacher_kwargs=None,
        is_regular_grid=True,
    ):
        """Collect data files from a dir to one output file.

        TODO: For a regular grid (lat values are constant across lon and vice
        versa) collecting lat / lon chunks is supported. For curvilinear grids
        only collection of chunks that are split by latitude are supported.
        This should be generalized to allow for any spatial chunking and any
        dimension. I think this would require a new file naming scheme with a
        spatial index for both latitude and longitude or checking each chunk
        to see how they are split.

        Filename requirements:
         - Should end with ".nc"

        Parameters
        ----------
        file_paths : list | str
            Explicit list of str file paths that will be sorted and collected
            or a single string with unix-style /search/patt*ern.nc.
        out_file : str
            File path of final output file.
        features : list | str
            List of dsets to collect. If 'all' then all ``data_vars`` will be
            collected.
        log_level : str | None
            Desired log level, None will not initialize logging.
        log_file : str | None
            Target log file. None logs to stdout.
        write_status : bool
            Flag to write status file once complete if running from pipeline.
        job_name : str
            Job name for status file if running from pipeline.
        overwrite : bool
            Whether to overwrite existing output file
        res_kwargs : dict | None
            Dictionary of kwargs to pass to xarray.open_mfdataset.
        cacher_kwargs : dict | None
            Dictionary of kwargs to pass to Cacher._write_single.
        is_regular_grid : bool
            Whether the data is on a regular grid. If True then spatial chunks
            can be combined across both latitude and longitude. If False then
            spatial chunks must all have the same longitude values to be
            combined. If you need completely general chunk collection then
            you should write chunks to `h5` files and use
            :class:`sup3r.postprocessing.collectors.h5.CollectorH5`.
        """
        logger.info(f'Initializing collection for file_paths={file_paths}')

        if log_level is not None:
            init_logger(
                'sup3r.preprocessing', log_file=log_file, log_level=log_level
            )

        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file), exist_ok=True)

        collector = cls(file_paths)
        logger.info(
            'Collecting {} files to {}'.format(len(collector.flist), out_file)
        )
        if overwrite and os.path.exists(out_file):
            logger.info(f'overwrite=True, removing {out_file}.')
            os.remove(out_file)

        spatial_chunks = collector.group_spatial_chunks()

        if not os.path.exists(out_file):
            res_kwargs = res_kwargs or {
                'combine': 'nested',
                'concat_dim': Dimension.TIME,
            }
            for s_idx, sfiles in spatial_chunks.items():
                schunk = Loader(sfiles, res_kwargs=res_kwargs)
                spatial_chunks[s_idx] = schunk

            # Set lat / lon as 1D arrays if regular grid and get the
            # xr.Dataset _ds
            if is_regular_grid:
                spatial_chunks = {
                    s_idx: schunk.set_regular_grid()._ds
                    for s_idx, schunk in spatial_chunks.items()
                }
                out = xr.combine_by_coords(
                    spatial_chunks.values(), combine_attrs='override'
                )

            else:
                out = xr.concat(
                    [sc._ds for sc in spatial_chunks.values()],
                    dim=Dimension.SOUTH_NORTH,
                )

            cacher_kwargs = cacher_kwargs or {}
            Cacher._write_single(
                out_file=out_file,
                data=out,
                features=features,
                **cacher_kwargs,
            )

        logger.info('Finished file collection.')

    def group_spatial_chunks(self):
        """Group same spatial chunks together so each entry has same spatial
        footprint but different times"""
        chunks = {}
        for file in self.flist:
            _, s_idx = self.get_chunk_indices(file)
            chunks[s_idx] = [*chunks.get(s_idx, []), file]
        for k, v in chunks.items():
            chunks[k] = sorted(v)
        return chunks
