"""H5 file collection."""

import logging
import os
from warnings import warn

import dask
import numpy as np
import pandas as pd
from rex.utilities.loggers import init_logger
from scipy.spatial import KDTree

from sup3r.preprocessing import Loader
from sup3r.preprocessing.utilities import _mem_check
from sup3r.utilities.utilities import get_dset_attrs, get_tmp_file
from sup3r.writers import RexOutputs

from .base import BaseCollector

logger = logging.getLogger(__name__)


class CollectorH5(BaseCollector):
    """Sup3r H5 file collection framework"""

    @classmethod
    def get_slices(
        cls, final_time_index, final_meta, new_time_index, new_meta
    ):
        """Get index slices where the new ti/meta belong in the final ti/meta.

        Parameters
        ----------
        final_time_index : pd.Datetimeindex
            Time index of the final file that new_time_index is being written
            to.
        final_meta : pd.DataFrame
            Meta data of the final file that new_meta is being written to.
        new_time_index : pd.Datetimeindex
            Chunk time index that is a subset of the final_time_index.
        new_meta : pd.DataFrame
            Chunk meta data that is a subset of the final_meta.

        Returns
        -------
        row_slice : slice
            final_time_index[row_slice] = new_time_index
        col_slice : slice
            final_meta[col_slice] = new_meta
        """
        final_index = final_meta.index
        new_index = new_meta.index
        row_loc = np.where(final_time_index.isin(new_time_index))[0]
        col_loc = np.where(final_meta['gid'].isin(new_meta['gid']))[0]

        if not len(row_loc) > 0:
            msg = (
                'Could not find row locations in file collection. '
                'New time index: {} final time index: {}'.format(
                    new_time_index, final_time_index
                )
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if not len(col_loc) > 0:
            msg = (
                'Could not find col locations in file collection. '
                'New index: {} final index: {}'.format(new_index, final_index)
            )
            logger.error(msg)
            raise RuntimeError(msg)

        row_slice = slice(np.min(row_loc), np.max(row_loc) + 1)

        msg = (
            f'row_slice={row_slice} conflict with row_indices={row_loc}. '
            'Indices do not seem to be increasing and/or contiguous.'
        )
        assert (row_slice.stop - row_slice.start) == len(row_loc), msg

        return row_slice, col_loc

    @classmethod
    def get_coordinate_indices(cls, target_meta, full_meta, threshold=1e-4):
        """Get coordindate indices in meta data for given targets

        Parameters
        ----------
        target_meta : pd.DataFrame
            Dataframe of coordinates to find within the full meta
        full_meta : pd.DataFrame
            Dataframe of full set of coordinates for unfiltered dataset
        threshold : float
            Threshold distance for finding target coordinates within full meta
        """
        ll2 = np.vstack((
            full_meta.latitude.values,
            full_meta.longitude.values,
        )).T
        tree = KDTree(ll2)
        targets = np.vstack((
            target_meta.latitude.values,
            target_meta.longitude.values,
        )).T
        _, indices = tree.query(targets, distance_upper_bound=threshold)
        indices = indices[indices < len(full_meta)]
        return indices

    def get_data(
        self,
        file_path,
        feature,
        time_index,
        meta,
        scale_factor,
        dtype,
        threshold=1e-4,
    ):
        """Retreive a data array from a chunked file.

        Parameters
        ----------
        file_path : str
            h5 file to get data from
        feature : str
            dataset to retrieve data from fpath.
        time_index : pd.Datetimeindex
            Time index of the final file.
        meta : pd.DataFrame
            Meta data of the final file.
        scale_factor : int | float
            Final destination scale factor after collection. If the data
            retrieval from the files to be collected has a different scale
            factor, the collected data will be rescaled and returned as
            float32.
        dtype : np.dtype
            Final dtype to return data as
        threshold : float
            Threshold distance for finding target coordinates within full meta

        Returns
        -------
        f_data : Union[np.ndarray, da.core.Array]
            Data array from the fpath cast as input dtype.
        row_slice : slice
            final_time_index[row_slice] = new_time_index
        col_slice : slice
            final_meta[col_slice] = new_meta
        """
        with RexOutputs(file_path, unscale=False, mode='r') as f:
            f_ti = f.time_index
            f_meta = f.meta

            if feature not in f.attrs:
                e = (
                    'Trying to collect dataset "{}" from {} but cannot find '
                    'in available attrbutes: {}'.format(
                        feature, file_path, f.attrs
                    )
                )
                logger.error(e)
                raise KeyError(e)

            source_scale_factor = f.attrs[feature].get('scale_factor', 1)

            if feature not in f.dsets:
                e = (
                    'Trying to collect dataset "{}" from {} but cannot find '
                    'in available features: {}'.format(
                        feature, file_path, f.dsets
                    )
                )
                logger.error(e)
                raise KeyError(e)

            mask = self.get_coordinate_indices(
                meta, f_meta, threshold=threshold
            )
            f_meta = f_meta.iloc[mask]
            f_data = f[feature][:, mask]

        if len(mask) == 0:
            msg = (
                'No target coordinates found in masked meta. '
                f'Skipping collection for {file_path}.'
            )
            logger.warning(msg)
            warn(msg)

        else:
            row_slice, col_slice = self.get_slices(
                time_index, meta, f_ti, f_meta
            )

            if scale_factor != source_scale_factor:
                f_data = f_data.astype(np.float32)
                f_data *= scale_factor / source_scale_factor

            if np.issubdtype(dtype, np.integer):
                f_data = np.round(f_data)

            f_data = f_data.astype(dtype)

            try:
                self.data[row_slice, col_slice] = f_data
            except Exception as e:
                msg = (
                    f'Failed to add data to self.data[{row_slice}, '
                    f'{col_slice}] for feature={feature}, '
                    f'file_path={file_path}, time_index={time_index}, '
                    f'meta={meta}. {e}'
                )
                logger.error(msg)
                raise OSError(msg) from e

    @classmethod
    def _get_file_time_index(cls, file):
        """Get time index for a single file. Simple method used in thread pool
        for attribute collection."""
        with RexOutputs(file, mode='r') as f:
            time_index = f.time_index
            logger.debug(
                'Finished getting time index for file: %s. %s',
                file,
                _mem_check(),
            )
        return time_index

    @classmethod
    def _get_file_meta(cls, file):
        """Get meta for a single file. Simple method used in thread pool for
        attribute collection."""
        with RexOutputs(file, mode='r') as f:
            meta = f.meta
            logger.debug(
                'Finished getting meta for file: %s. %s', file, _mem_check()
            )
        return meta

    def get_unique_chunk_files(self, file_paths):
        """We get files for the unique spatial and temporal extents covered by
        all collection files. Since the files have a suffix
        ``_{temporal_chunk_index}_{spatial_chunk_index}.h5`` we just use all
        files with a single ``spatial_chunk_index`` for the full time index and
        all files with a single ``temporal_chunk_index`` for the full meta.

        Parameters
        ----------
        t_files : list
            Explicit list of str file paths which, when combined, provide the
            entire spatial domain.
        s_files : list
            Explicit list of str file paths which, when combined, provide the
            entire temporal extent.
        """
        t_files = {}
        s_files = {}
        for f in file_paths:
            t_chunk, s_chunk = self.get_chunk_indices(f)
            if t_chunk not in t_files:
                t_files[t_chunk] = f
            if s_chunk not in s_files:
                s_files[s_chunk] = f

        t_files = list(t_files.values())
        s_files = list(s_files.values())
        logger.info('Found %s unique temporal chunks', len(t_files))
        logger.info('Found %s unique spatial chunks', len(s_files))
        return t_files, s_files

    def _get_collection_attrs(self, file_paths, max_workers=None):
        """Get important dataset attributes from a file list to be collected.

        Assumes the file list is chunked in time (row chunked).

        Parameters
        ----------
        file_paths : list | str
            Explicit list of str file paths that will be sorted and collected
            or a single string with unix-style /search/patt*ern.h5.
        max_workers : int | None
            Number of workers to use in parallel. 1 runs serial,
            None uses all available.

        Returns
        -------
        time_index : pd.datetimeindex
            Concatenated full size datetime index from the flist that is
            being collected
        meta : pd.DataFrame
            Concatenated full size meta data from the flist that is being
            collected or provided target meta
        """

        t_files, s_files = self.get_unique_chunk_files(file_paths)
        meta_tasks = [dask.delayed(self._get_file_meta)(fn) for fn in s_files]
        ti_tasks = [
            dask.delayed(self._get_file_time_index)(fn) for fn in t_files
        ]

        if max_workers == 1:
            meta = dask.compute(*meta_tasks, scheduler='single-threaded')
            time_index = dask.compute(*ti_tasks, scheduler='single-threaded')
        else:
            meta = dask.compute(
                *meta_tasks, scheduler='threads', num_workers=max_workers
            )
            time_index = dask.compute(
                *ti_tasks, scheduler='threads', num_workers=max_workers
            )
        logger.info(
            'Finished getting meta and time_index for all unique chunks.'
        )
        time_index = pd.DatetimeIndex(np.concatenate(time_index))
        time_index = time_index.sort_values()
        unique_ti = time_index.drop_duplicates()
        msg = 'Found duplicate time steps from supposedly unique time periods.'
        assert len(unique_ti) == len(time_index), msg
        meta = pd.concat(meta)

        if 'latitude' in meta and 'longitude' in meta:
            meta = meta.drop_duplicates(subset=['latitude', 'longitude'])
        meta = meta.sort_values('gid')

        logger.info('Finished building full meta and time index.')
        return time_index, meta

    def get_target_and_masked_meta(
        self, meta, target_meta_file=None, threshold=1e-4
    ):
        """Use combined meta for all files and target_meta_file to get
        mapping from the full meta to the target meta and the mapping from the
        target meta to the full meta, both of which are masked to remove
        coordinates not present in the target_meta.

        Parameters
        ----------
        meta : pd.DataFrame
            Concatenated full size meta data from the flist that is being
            collected or provided target meta
        target_meta_file : str
            Path to target final meta containing coordinates to keep from the
            full list of coordinates present in the collected meta for the full
            file list.
        threshold : float
            Threshold distance for finding target coordinates within full meta

        Returns
        -------
        target_meta : pd.DataFrame
            Concatenated full size meta data from the flist that is being
            collected or provided target meta
        masked_meta : pd.DataFrame
            Concatenated full size meta data from the flist that is being
            collected masked against target_meta
        """
        if target_meta_file is not None and os.path.exists(target_meta_file):
            target_meta = pd.read_csv(target_meta_file)
            if 'gid' in target_meta.columns:
                target_meta = target_meta.drop('gid', axis=1)
            mask = self.get_coordinate_indices(
                target_meta, meta, threshold=threshold
            )
            masked_meta = meta.iloc[mask]
            logger.info(f'Masked meta coordinates: {len(masked_meta)}')
            mask = self.get_coordinate_indices(
                masked_meta, target_meta, threshold=threshold
            )
            target_meta = target_meta.iloc[mask]
            logger.info(f'Target meta coordinates: {len(target_meta)}')
        else:
            target_meta = masked_meta = meta

        return target_meta, masked_meta

    def get_collection_attrs(
        self,
        file_paths,
        max_workers=None,
        target_meta_file=None,
        threshold=1e-4,
    ):
        """Get important dataset attributes from a file list to be collected.

        Assumes the file list is chunked in time (row chunked).

        Parameters
        ----------
        file_paths : list | str
            Explicit list of str file paths that will be sorted and collected
            or a single string with unix-style /search/patt*ern.h5.
        max_workers : int | None
            Number of workers to use in parallel. 1 runs serial,
            None will use all available workers.
        target_meta_file : str
            Path to target final meta containing coordinates to keep from the
            full list of coordinates present in the collected meta for the full
            file list.
        threshold : float
            Threshold distance for finding target coordinates within full meta

        Returns
        -------
        time_index : pd.datetimeindex
            Concatenated full size datetime index from the flist that is
            being collected
        target_meta : pd.DataFrame
            Concatenated full size meta data from the flist that is being
            collected or provided target meta
        masked_meta : pd.DataFrame
            Concatenated full size meta data from the flist that is being
            collected masked against target_meta
        shape : tuple
            Output (collected) dataset shape
        global_attrs : dict
            Global attributes from the first file in file_paths (it's assumed
            that all the files in file_paths have the same global file
            attributes).
        """
        logger.info(f'Using target_meta_file={target_meta_file}')
        if isinstance(target_meta_file, str):
            msg = f'Provided target meta ({target_meta_file}) does not exist.'
            assert os.path.exists(target_meta_file), msg

        time_index, meta = self._get_collection_attrs(
            file_paths, max_workers=max_workers
        )
        logger.info('Getting target and masked meta.')
        target_meta, masked_meta = self.get_target_and_masked_meta(
            meta, target_meta_file, threshold=threshold
        )

        shape = (len(time_index), len(target_meta))

        logger.info('Getting global attrs from %s', file_paths[0])
        with RexOutputs(file_paths[0], mode='r') as fin:
            global_attrs = fin.global_attrs

        return time_index, target_meta, masked_meta, shape, global_attrs

    def _write_flist_data(
        self,
        out_file,
        feature,
        time_index,
        subset_masked_meta,
        target_masked_meta,
    ):
        """Write spatiotemporal file list data to output file for given
        feature

        Parameters
        ----------
        out_file : str
            Name of output file
        feature : str
            Name of feature for output chunk
        time_index : pd.DateTimeIndex
            Time index for corresponding file list data
        subset_masked_meta : pd.DataFrame
            Meta for corresponding file list data
        target_masked_meta : pd.DataFrame
            Meta for full output file
        """
        with RexOutputs(out_file, mode='r') as f:
            target_ti = f.time_index
            y_write_slice, x_write_slice = self.get_slices(
                target_ti,
                target_masked_meta,
                time_index,
                subset_masked_meta,
            )
        self._ensure_dset_in_output(out_file, feature)

        with RexOutputs(out_file, mode='a') as f:
            try:
                f[feature, y_write_slice, x_write_slice] = self.data
            except Exception as e:
                msg = (
                    f'Problem with writing data to {out_file} with '
                    f't_slice={y_write_slice}, '
                    f's_slice={x_write_slice}. {e}'
                )
                logger.error(msg)
                raise OSError(msg) from e

        logger.debug(
            'Finished writing "{}" for row {} and col {} to: {}'.format(
                feature,
                y_write_slice,
                x_write_slice,
                os.path.basename(out_file),
            )
        )

    def _collect_flist(
        self,
        feature,
        subset_masked_meta,
        time_index,
        shape,
        file_paths,
        out_file,
        target_masked_meta,
        max_workers=None,
    ):
        """Collect a dataset from a file list without getting attributes first.
        This file list can be a subset of a full file list to be collected.

        Parameters
        ----------
        feature : str
            Dataset name to collect.
        subset_masked_meta : pd.DataFrame
            Meta data containing the list of coordinates present in both the
            given file paths and the target_meta. This can be a subset of
            the coordinates present in the full file list. The coordinates
            contained in this dataframe have the same gids as those present in
            the meta for the full file list.
        time_index : pd.datetimeindex
            Concatenated datetime index for the given file paths.
        shape : tuple
            Output (collected) dataset shape
        file_paths : list | str
            File list to be collected. This can be a subset of a full file list
            to be collected.
        out_file : str
            File path of final output file.
        target_masked_meta : pd.DataFrame
            Same as subset_masked_meta but instead for the entire list of files
            to be collected.
        max_workers : int | None
            Number of workers to use in parallel. 1 runs serial,
            None uses all available.
        """
        if len(subset_masked_meta) > 0:
            attrs, final_dtype = get_dset_attrs(feature)
            scale_factor = attrs.get('scale_factor', 1)

            logger.debug(
                'Collecting file list of shape %s: %s', shape, file_paths
            )

            self.data = np.zeros(shape, dtype=final_dtype)
            logger.debug(
                'Initializing output dataset "%s" in-memory with '
                'shape %s and dtype %s. %s',
                feature,
                shape,
                final_dtype,
                _mem_check(),
            )
            tasks = []
            for fname in file_paths:
                task = dask.delayed(self.get_data)(
                    fname,
                    feature,
                    time_index,
                    subset_masked_meta,
                    scale_factor,
                    final_dtype,
                )
                tasks.append(task)
            if max_workers == 1:
                logger.info(
                    'Running serial collection on %s files', len(file_paths)
                )
                dask.compute(*tasks, scheduler='single-threaded')
            else:
                logger.info(
                    'Running parallel collection on %s files with '
                    'max_workers=%s.',
                    len(file_paths),
                    max_workers,
                )
                dask.compute(
                    *tasks, scheduler='threads', num_workers=max_workers
                )
            logger.info('Finished collection of %s files.', len(file_paths))
            self._write_flist_data(
                out_file,
                feature,
                time_index,
                subset_masked_meta,
                target_masked_meta,
            )
        else:
            msg = (
                'No target coordinates found in masked meta. Skipping '
                f'collection for {file_paths}.'
            )
            logger.warning(msg)
            warn(msg)

    def get_flist_chunks(self, file_paths, n_writes=None):
        """Group files by temporal_chunk_index and then combines these groups
        if ``n_writes`` is less than the number of time_chunks. Assumes
        file_paths have a suffix format like
        ``_{temporal_chunk_index}_{spatial_chunk_index}.h5``

        Parameters
        ----------
        file_paths : list
            List of file paths each with a suffix
            ``_{temporal_chunk_index}_{spatial_chunk_index}.h5``
        n_writes : int | None
            Number of writes to use for collection

        Returns
        -------
        flist_chunks : list
            List of file list chunks. Used to split collection and writing into
            multiple steps.
        """
        file_chunks = {}
        for file in file_paths:
            t_chunk, _ = self.get_chunk_indices(file)
            file_chunks[t_chunk] = [*file_chunks.get(t_chunk, []), file]

        if n_writes is not None and n_writes > len(file_chunks):
            logger.info(
                f'n_writes ({n_writes}) too big, setting to the number '
                f'of temporal chunks ({len(file_chunks)}).'
            )
            n_writes = len(file_chunks)

        n_writes = n_writes or len(file_chunks)
        tc_groups = np.array_split(list(file_chunks.keys()), n_writes)
        fp_groups = [[file_chunks[tc] for tc in tcs] for tcs in tc_groups]
        flist_chunks = [np.concatenate(group) for group in fp_groups]
        logger.debug(
            'Split file list into %s chunks according to n_writes=%s',
            len(flist_chunks),
            n_writes,
        )

        logger.debug(f'Grouped file list into {len(file_chunks)} time chunks.')

        return flist_chunks

    def collect_feature(
        self,
        dset,
        target_masked_meta,
        target_meta_file,
        time_index,
        shape,
        flist_chunks,
        out_file,
        threshold=1e-4,
        max_workers=None,
    ):
        """Collect chunks for single feature

        dset : str
            Dataset name to collect.
        target_masked_meta : pd.DataFrame
            Same as subset_masked_meta but instead for the entire list of files
            to be collected.
        target_meta_file : str
            Path to target final meta containing coordinates to keep from the
            full file list collected meta. This can be but is not necessarily a
            subset of the full list of coordinates for all files in the file
            list. This is used to remove coordinates from the full file list
            which are not present in the target_meta. Either this full
            meta or a subset, depending on which coordinates are present in
            the data to be collected, will be the final meta for the collected
            output files.
        time_index : pd.datetimeindex
            Concatenated datetime index for the given file paths.
        shape : tuple
            Output (collected) dataset shape
        flist_chunks : list
            List of file list chunks. Used to split collection and writing into
            multiple steps.
        out_file : str
            File path of final output file.
        threshold : float
            Threshold distance for finding target coordinates within full meta
        max_workers : int | None
            Number of workers to use in parallel. 1 runs serial,
            None will use all available workers.
        """
        logger.debug('Collecting dataset "%s".', dset)

        if len(flist_chunks) == 1:
            self._collect_flist(
                dset,
                target_masked_meta,
                time_index,
                shape,
                flist_chunks[0],
                out_file,
                target_masked_meta,
                max_workers=max_workers,
            )

        else:
            for i, flist in enumerate(flist_chunks):
                logger.info(
                    'Collecting file list chunk %s out of %s ',
                    i + 1,
                    len(flist_chunks),
                )
                out = self.get_collection_attrs(
                    flist,
                    max_workers=max_workers,
                    target_meta_file=target_meta_file,
                    threshold=threshold,
                )
                time_index, _, masked_meta, shape, _ = out
                self._collect_flist(
                    dset,
                    masked_meta,
                    time_index,
                    shape,
                    flist,
                    out_file,
                    target_masked_meta,
                    max_workers=max_workers,
                )

    @classmethod
    def collect(
        cls,
        file_paths,
        out_file,
        features='all',
        max_workers=None,
        log_level=None,
        log_file=None,
        target_meta_file=None,
        n_writes=None,
        overwrite=True,
        threshold=1e-4,
    ):
        """Collect data files from a dir to one output file.

        Filename requirements:
         - Should end with ".h5"

        Parameters
        ----------
        file_paths : list | str
            Explicit list of str file paths that will be sorted and collected
            or a single string with unix-style /search/patt*ern.h5. Files
            resolved by this argument must be of the form
            ``*_{temporal_chunk_index}_{spatial_chunk_index}.h5``.
        out_file : str
            File path of final output file.
        features : list
            List of dsets to collect
        max_workers : int | None
            Number of workers to use in parallel. 1 runs serial,
            None will use all available workers.
        log_level : str | None
            Desired log level, None will not initialize logging.
        log_file : str | None
            Target log file. None logs to stdout.
        write_status : bool
            Flag to write status file once complete if running from pipeline.
        job_name : str
            Job name for status file if running from pipeline.
        pipeline_step : str, optional
            Name of the pipeline step being run. If ``None``, the
            ``pipeline_step`` will be set to ``"collect``, mimicking old reV
            behavior. By default, ``None``.
        target_meta_file : str
            Path to target final meta containing coordinates to keep from the
            full file list collected meta. This can be but is not necessarily a
            subset of the full list of coordinates for all files in the file
            list. This is used to remove coordinates from the full file list
            which are not present in the target_meta. Either this full
            meta or a subset, depending on which coordinates are present in
            the data to be collected, will be the final meta for the collected
            output files.
        n_writes : int | None
            Number of writes to split full file list into. Must be less than
            or equal to the number of temporal chunks if chunks have different
            time indices.
        overwrite : bool
            Whether to overwrite existing output file
        threshold : float
            Threshold distance for finding target coordinates within full meta
        """
        logger.info(
            'Initializing collection for file_paths=%s with max_workers=%s',
            file_paths,
            max_workers,
        )

        if log_level is not None:
            init_logger(
                'sup3r.preprocessing', log_file=log_file, log_level=log_level
            )

        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file), exist_ok=True)

        collector = cls(file_paths)
        features = (
            Loader(collector.flist[0]).features
            if features == 'all'
            else features
        )
        logger.info(
            'Collecting %s files to %s', len(collector.flist), out_file
        )
        if overwrite and os.path.exists(out_file):
            logger.info('overwrite=True, removing %s', out_file)
            os.remove(out_file)

        out = collector.get_collection_attrs(
            collector.flist,
            max_workers=max_workers,
            target_meta_file=target_meta_file,
            threshold=threshold,
        )
        logger.info('Finished building full spatiotemporal collection extent.')
        time_index, target_meta, target_masked_meta = out[:3]
        shape, global_attrs = out[3:]

        flist_chunks = collector.get_flist_chunks(
            collector.flist, n_writes=n_writes
        )
        tmp_file = get_tmp_file(out_file)
        if not os.path.exists(out_file):
            collector._init_h5(tmp_file, time_index, target_meta, global_attrs)
        for dset in features:
            logger.debug('Collecting dataset "%s".', dset)
            collector.collect_feature(
                dset=dset,
                target_masked_meta=target_masked_meta,
                target_meta_file=target_meta_file,
                time_index=time_index,
                shape=shape,
                flist_chunks=flist_chunks,
                out_file=tmp_file,
                threshold=threshold,
                max_workers=max_workers,
            )
        os.replace(tmp_file, out_file)
        logger.info('Finished file collection.')
