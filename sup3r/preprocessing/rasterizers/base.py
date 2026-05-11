"""Objects that can rasterize 3D spatiotemporal data. Examples include WRF,
ERA5, and GCM data. Can also work with 3D H5 data, just not flattened H5 data
like WTK and NSRDB."""

import logging
from warnings import warn

import numpy as np

from sup3r.preprocessing.base import Container
from sup3r.preprocessing.names import Dimension
from sup3r.preprocessing.utilities import _parse_time_slice, compute_if_dask

logger = logging.getLogger(__name__)


class BaseRasterizer(Container):
    """Container subclass with additional methods for extracting a
    spatiotemporal extent from contained data.

    Note
    ----
    This `Rasterizer` base class is for 3D rasterized data. This usually
    comes from NETCDF files but can also be cached H5 files saved from
    previously rasterized data. For 3D, whether H5 or NETCDF, the full domain
    will be rasterized automatically if no target / shape are provided."""

    def __init__(
        self,
        loader,
        features='all',
        target=None,
        shape=None,
        time_slice=slice(None),
        threshold=None,
    ):
        """
        Parameters
        ----------
        loader : Loader
            Loader type container with `.data` attribute exposing data to
            extract.
        features : list | str
            Features to return in loaded dataset. If 'all' then all available
            features will be returned.
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape or
            raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        time_slice : slice | list
            Slice specifying extent and step of temporal extraction. e.g.
            slice(start, stop, step). If equal to slice(None, None, 1) the full
            time dimension is selected. Can be also be a list ``[start, stop,
            step]``
        threshold : float
            Nearest neighbor euclidean distance threshold. If the coordinates
            are more than this value away from the target lat/lon, an error is
            raised.
        """
        logger.debug(
            'Rasterizing features: "%s" from files: %s',
            features,
            loader.file_paths,
        )
        super().__init__(data=loader.data)
        self.loader = loader
        self.threshold = threshold
        self.time_slice = time_slice
        self.grid_shape = shape
        self.target = target
        self.full_lat_lon = self.data.lat_lon
        self.raster_index = self.get_raster_index()
        self.time_index = (
            loader.time_index[self.time_slice]
            if 'time' in loader.data.indexes
            else None
        )
        self._lat_lon = None
        self.data = self.rasterize_data()
        self.data = self.data[features]

    @property
    def time_slice(self):
        """Return time slice for rasterized time period."""
        return self._time_slice

    @time_slice.setter
    def time_slice(self, value):
        """Set and sanitize the time slice."""
        self._time_slice = _parse_time_slice(value)

    @property
    def target(self):
        """Return the true value based on the closest lat lon instead of the
        user provided value self._target, which is used to find the closest lat
        lon."""
        return np.asarray(self.lat_lon[-1, 0])

    @target.setter
    def target(self, value):
        """Set the private target attribute. Ultimately target is determined by
        lat_lon but _target is set to bottom left corner of the full domain if
        None and then used to get the raster_index, which is then used to get
        the lat_lon"""
        self._target = np.asarray(value) if value is not None else None

    @property
    def grid_shape(self):
        """Return the grid_shape based on the raster_index, since
        self._grid_shape does not need to be provided as an input if the
        raster_file is."""
        return self.lat_lon.shape[:-1]

    @grid_shape.setter
    def grid_shape(self, value):
        """Set the private grid_shape attribute. Ultimately grid_shape is
        determined by lat_lon but _grid_shape is set to the full domain if None
        and then used to get the raster_index, which is then used to get the
        lat_lon"""
        self._grid_shape = value

    @property
    def lat_lon(self):
        """Get 2D grid of coordinates with `target` as the lower left
        coordinate. (lats, lons, 2)"""
        if self._lat_lon is None:
            self._lat_lon = self.get_lat_lon()
        return self._lat_lon

    def rasterize_data(self):
        """Get rasterized data."""
        logger.debug(
            'Rasterizing data for target / shape: %s / %s',
            np.asarray(self._target),
            np.asarray(self._grid_shape),
        )
        kwargs = dict(zip(Dimension.dims_2d(), self.raster_index))
        if Dimension.TIME in self.loader.dims:
            kwargs[Dimension.TIME] = self.time_slice
        return self.loader.isel(**kwargs)

    def check_target_and_shape(self, full_lat_lon):
        """The data is assumed to use a regular grid so if either target or
        shape is not given we can easily find the values that give the maximum
        extent."""
        if self._target is None:
            self._target = full_lat_lon[-1, 0, :]
        if self._grid_shape is None:
            self._grid_shape = full_lat_lon.shape[:-1]

    def get_raster_index(self):
        """Get set of slices or indices selecting the requested region from
        the contained data."""
        logger.debug(
            'Getting raster index for target / shape: %s / %s',
            np.asarray(self._target),
            np.asarray(self._grid_shape),
        )
        self.check_target_and_shape(self.full_lat_lon)
        row, col = self.get_closest_row_col(self.full_lat_lon, self._target)
        lat_slice = slice(row - self._grid_shape[0] + 1, row + 1)
        lon_slice = slice(col, col + self._grid_shape[1])
        return self._check_raster_index(lat_slice, lon_slice)

    def _check_raster_index(self, lat_slice, lon_slice):
        """Check if raster index has bounds which exceed available region and
        crop if so."""
        lat_start, lat_end = lat_slice.start, lat_slice.stop
        lon_start, lon_end = lon_slice.start, lon_slice.stop
        lat_start = max(lat_start, 0)
        lat_end = min(lat_end, self.full_lat_lon.shape[0])
        lon_start = max(lon_start, 0)
        lon_end = min(lon_end, self.full_lat_lon.shape[1])
        new_lat_slice = slice(lat_start, lat_end)
        new_lon_slice = slice(lon_start, lon_end)
        msg = (
            f'Computed lat_slice = {compute_if_dask(lat_slice)} exceeds '
            f'available region. Using {compute_if_dask(new_lat_slice)}.'
        )
        if lat_slice != new_lat_slice:
            logger.warning(msg)
            warn(msg)
        msg = (
            f'Computed lon_slice = {compute_if_dask(lon_slice)} exceeds '
            f'available region. Using {compute_if_dask(new_lon_slice)}.'
        )
        if lon_slice != new_lon_slice:
            logger.warning(msg)
            warn(msg)
        return new_lat_slice, new_lon_slice

    def get_closest_row_col(self, lat_lon, target):
        """Get closest indices to target lat lon

        Parameters
        ----------
        lat_lon : ndarray
            Array of lat/lon
            (spatial_1, spatial_2, 2)
            Last dimension in order of (lat, lon)
        target : tuple
            (lat, lon) for target coordinate

        Returns
        -------
        row : int
            row index for closest lat/lon to target lat/lon
        col : int
            col index for closest lat/lon to target lat/lon
        """
        dist = np.hypot(
            lat_lon[..., 0] - target[0], lat_lon[..., 1] - target[1]
        )
        row, col = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        closest_coord = np.asarray(lat_lon[row, col])
        requested_target = np.asarray(target)
        min_dist = np.asarray(dist.min())
        if self.threshold is not None and dist.min() > self.threshold:
            logger.error(
                'The distance between the closest coordinate: %s and the '
                'requested target: %s is %s. This exceeds the given '
                'threshold: %s',
                closest_coord,
                requested_target,
                min_dist,
                self.threshold,
            )
            raise RuntimeError(
                'The distance between the closest coordinate: {} and the '
                'requested target: {} is {}. This exceeds the given '
                'threshold: {}'.format(
                    closest_coord,
                    requested_target,
                    min_dist,
                    self.threshold,
                )
            )
        logger.debug(
            'The distance between the closest coordinate: %s and the '
            'requested target: %s is %s.',
            closest_coord,
            requested_target,
            min_dist,
        )
        return row, col

    def get_lat_lon(self):
        """Get the 2D array of coordinates corresponding to the requested
        target and shape."""
        return self.full_lat_lon[self.raster_index[0], self.raster_index[1]]
