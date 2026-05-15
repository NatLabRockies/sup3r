"""Utilities used for QA"""
import logging

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def tke_frequency_spectrum(u, v, f_range=None):
    """Kinetic Energy Spectrum. Gives the portion of kinetic energy
    associated with each frequency.

    Parameters
    ----------
    u: ndarray
        (lat, lon)
        U component of wind
    v : ndarray
        (lat, lon)
        V component of wind
    f_range : list | None
        List with min and max frequency. When comparing spectra for different
        domains this needs to be tailored to the specific domain.  e.g. f =
        [1/max_time, ..., 1/min_time] If this is not specified f with be
        set to [0, ..., len(y)] where y is the fft output.

    Returns
    -------
    ndarray
        1D array of amplitudes corresponding to the portion of total energy
        with a given frequency
    """
    v_f = np.fft.fftn(v.reshape((-1, v.shape[-1])))
    u_f = np.fft.fftn(u.reshape((-1, u.shape[-1])))
    E_f = np.abs(v_f)**2 + np.abs(u_f)**2
    E_f = np.mean(E_f, axis=0)
    if f_range is None:
        f = np.arange(len(E_f))
    else:
        f = np.linspace(f_range[0], f_range[1], len(E_f))
    E_f = f**2 * E_f
    n_steps = E_f.shape[0] // 2
    E_f_a = E_f[:n_steps]
    E_f_b = E_f[-n_steps:][::-1]
    E_f = E_f_a + E_f_b
    return f[:n_steps], E_f


def frequency_spectrum(var, f_range=None):
    """Frequency Spectrum. Gives the portion of the variable
    associated with each frequency.

    Parameters
    ----------
    var: ndarray
        (lat, lon, temporal)
    f_range : list | None
        List with min and max frequency. When comparing spectra for different
        domains this needs to be tailored to the specific domain.  e.g. f =
        [1/max_time, ..., 1/min_time] If this is not specified f with be
        set to [0, ..., len(y)] where y is the fft output.

    Returns
    -------
    ndarray
        Array of frequencies corresponding to energy amplitudes
    ndarray
        1D array of amplitudes corresponding to the portion of the variable
        with a given frequency
    """
    var_f = np.fft.fftn(var.reshape((-1, var.shape[-1])))
    E_f = np.abs(var_f)**2
    E_f = np.mean(E_f, axis=0)
    if f_range is None:
        f = np.arange(len(E_f))
    else:
        f = np.linspace(f_range[0], f_range[1], len(E_f))
    E_f = f**2 * E_f
    n_steps = E_f.shape[0] // 2
    E_f_a = E_f[:n_steps]
    E_f_b = E_f[-n_steps:][::-1]
    E_f = E_f_a + E_f_b
    return f[:n_steps], E_f


def tke_wavenumber_spectrum(u, v, x_range=None, axis=0):
    """Turbulent Kinetic Energy Spectrum. Gives the portion of kinetic energy
    associated with each wavenumber.

    Parameters
    ----------
    u: ndarray
        (lat, lon)
        U component of wind
    v : ndarray
        (lat, lon)
        V component of wind
    x_range : list | None
        List with min and max wavenumber. When comparing spectra for different
        domains this needs to be tailored to the specific domain.  e.g. k =
        [1/max_length, ..., 1/min_length] If this is not specified k with be
        set to [0, ..., len(y)] where y is the fft output.
    axis : int
        Axis to average over to get a 1D wind field. If axis=0 this returns
        the zonal energy spectrum

    Returns
    -------
    ndarray
        Array of wavenumbers corresponding to energy amplitudes
    ndarray
        1D array of amplitudes corresponding to the portion of total energy
        with a given wavenumber
    """
    u_k = np.fft.fftn(u)
    v_k = np.fft.fftn(v)
    E_k = np.mean(np.abs(v_k)**2 + np.abs(u_k)**2, axis=axis)
    if x_range is None:
        k = np.arange(len(E_k))
    else:
        k = np.linspace(x_range[0], x_range[1], len(E_k))
    n_steps = len(k) // 2
    E_k = k**2 * E_k
    E_k_a = E_k[1:n_steps + 1]
    E_k_b = E_k[-n_steps:][::-1]
    E_k = E_k_a + E_k_b
    return k[:n_steps], E_k


def wavenumber_spectrum(var, x_range=None, axis=0):
    """Wavenumber Spectrum. Gives the portion of the given variable
    associated with each wavenumber.

    Parameters
    ----------
    var: ndarray
        (lat, lon)
    x_range : list | None
        List with min and max wavenumber. When comparing spectra for different
        domains this needs to be tailored to the specific domain.  e.g. k =
        [1/max_length, ..., 1/min_length] If this is not specified k with be
        set to [0, ..., len(y)] where y is the fft output.
    axis : int
        Axis to average over to get a 1D field. If axis=0 this returns
        the zonal spectrum

    Returns
    -------
    ndarray
        Array of wavenumbers corresponding to amplitudes
    ndarray
        1D array of amplitudes corresponding to the portion of the given
        variable with a given wavenumber
    """
    var_k = np.fft.fftn(var)
    E_k = np.mean(np.abs(var_k)**2, axis=axis)
    if x_range is None:
        k = np.arange(len(E_k))
    else:
        k = np.linspace(x_range[0], x_range[1], len(E_k))
    n_steps = len(k) // 2
    E_k = k**2 * E_k
    E_k_a = E_k[1:n_steps + 1]
    E_k_b = E_k[-n_steps:][::-1]
    E_k = E_k_a + E_k_b
    return k[:n_steps], E_k


def direct_dist(var, bins=40, range=None, diff_max=None, scale=1,
                percentile=99.9, interpolate=False, period=None):
    """Returns the direct distribution for the given variable.

    Parameters
    ----------
    var: ndarray
        (lat, lon, temporal)
    bins : int
        Number of bins for the direct pdf.
    range : tuple | None
        Optional min/max range for the direct pdf.
    diff_max : float
        Max value to keep for given variable
    scale : int
        Factor to scale the distribution by. This is used so that distributions
        from data with different resolutions can be compared. For instance, if
        this is calculating a vorticity distribution from data with a spatial
        resolution of 4km then the distribution needs to be scaled by 4km to
        compare to another scaled vorticity distribution with a different
        resolution.
    percentile : float
        Percentile to use to determine the maximum allowable value in the
        distribution. e.g. percentile=99 eliminates values above the 99th
        percentile from the histogram.
    interpolate : bool
        Whether to interpolate over histogram counts. e.g. if a bin has
        count = 0 and surrounding bins have count > 0 the bin with count = 0
        will have an interpolated value.
    period : float | None
        If variable is periodic this gives that period. e.g. If the variable
        is winddirection the period is 360 degrees and we need to account for
        0 and 360 being close.

    Returns
    -------
    ndarray
        var at bin centers
    ndarray
        var value counts
    float
        Normalization factor
    """

    if period is not None:
        diffs = (var + period) % period
        diffs /= scale
    else:
        diffs = var / scale
    diff_max = diff_max or np.percentile(np.abs(diffs), percentile)
    diffs = diffs[(np.abs(diffs) < diff_max)]
    norm = np.sqrt(np.mean(diffs**2))
    counts, centers = continuous_dist(diffs, bins=bins, range=range,
                                      interpolate=interpolate)
    return centers, counts, norm


def gradient_dist(var, bins=40, range=None, diff_max=None, scale=1,
                  percentile=99.9, interpolate=False, period=None):
    """Returns the gradient distribution for the given variable.

    Parameters
    ----------
    var: ndarray
        (lat, lon, temporal)
    bins : int
        Number of bins for the gradient pdf.
    range : tuple | None
        Optional min/max range for the gradient pdf.
    diff_max : float
        Max value to keep for gradient
    scale : int
        Factor to scale the distribution by. This is used so that distributions
        from data with different resolutions can be compared. For instance, if
        this is calculating a velocity gradient distribution from data with a
        spatial resolution of 4km then the distribution needs to be scaled by
        4km to compare to another scaled velocity gradient distribution with a
        different resolution.
    percentile : float
        Percentile to use to determine the maximum allowable value in the
        distribution. e.g. percentile=99 eliminates values above the 99th
        percentile from the histogram.
    interpolate : bool
        Whether to interpolate over histogram counts. e.g. if a bin has
        count = 0 and surrounding bins have count > 0 the bin with count = 0
        will have an interpolated value.
    period : float | None
        If variable is periodic this gives that period. e.g. If the variable
        is winddirection the period is 360 degrees and we need to account for
        0 and 360 being close.

    Returns
    -------
    ndarray
        d(var) / dx at bin centers
    ndarray
        d(var) / dx value counts
    float
        Normalization factor
    """
    diffs = np.diff(var, axis=1).flatten()
    if period is not None:
        diffs = (diffs + period / 2) % period - period / 2
    diffs /= scale
    diff_max = diff_max or np.percentile(np.abs(diffs), percentile)
    diffs = diffs[(np.abs(diffs) < diff_max)]
    norm = np.sqrt(np.mean(diffs**2))
    counts, centers = continuous_dist(diffs, bins=bins, range=range,
                                      interpolate=interpolate)
    return centers, counts, norm


def time_derivative_dist(var, bins=40, range=None, diff_max=None, t_steps=1,
                         scale=1, percentile=99.9, interpolate=False,
                         period=None):
    """Returns the time derivative distribution for the given variable.

    Parameters
    ----------
    var: ndarray
        (lat, lon, temporal)
    bins : int
        Number of bins for the time derivative pdf.
    range : tuple | None
        Optional min/max range for the time derivative pdf.
    diff_max : float
        Max value to keep for time derivative
    t_steps : int
        Number of time steps to use for differences. e.g. If t_steps=1 this
        uses var[i + 1] - [i] to compute time derivatives.
    scale : int
        Factor to scale the distribution by. This is used so that distributions
        from data with different resolutions can be compared. For instance, if
        this is calculating a time derivative distribution from data with a
        temporal resolution of 15min then the distribution needs to be scaled
        by 15min to compare to another scaled time derivative distribution with
        a different resolution
    percentile : float
        Percentile to use to determine the maximum allowable value in the
        distribution. e.g. percentile=99 eliminates values above the 99th
        percentile from the histogram.
    interpolate : bool
        Whether to interpolate over histogram counts. e.g. if a bin has
        count = 0 and surrounding bins have count > 0 the bin with count = 0
        will have an interpolated value.
    period : float | None
        If variable is periodic this gives that period. e.g. If the variable
        is winddirection the period is 360 degrees and we need to account for
        0 and 360 being close.

    Returns
    -------
    ndarray
        d(var) / dt values at bin centers
    ndarray
        d(var) / dt value counts
    float
        Normalization factor
    """

    msg = (f'Received t_steps={t_steps} for time derivative calculation but '
           'data only has {var.shape[-1]} time steps')
    assert t_steps < var.shape[-1], msg
    diffs = (var[..., t_steps:] - var[..., :-t_steps]).flatten()
    if period is not None:
        diffs = (diffs + period / 2) % period - period / 2
    diffs /= scale
    diff_max = diff_max or np.percentile(np.abs(diffs), percentile)
    diffs = diffs[(np.abs(diffs) < diff_max)]
    norm = np.sqrt(np.mean(diffs**2))
    counts, centers = continuous_dist(diffs, bins=bins, range=range,
                                      interpolate=interpolate)
    return centers, counts, norm


def continuous_dist(diffs, bins=None, range=None, interpolate=False):
    """Get interpolated distribution from histogram

    Parameters
    ----------
    diffs : ndarray
        Array of values to use to construct distribution
    bins : int
        Number of bins for the distribution. If None then the number of bins
        will be determined from the value range and the smallest difference
        between values
    range : tuple | None
        Optional min/max range for the distribution.
    interpolate : bool
        Whether to interpolate over histogram counts. e.g. if a bin has
        count = 0 and surrounding bins have count > 0 the bin with count = 0
        will have an interpolated value.

    Returns
    -------
    ndarray
        distribution value counts
    ndarray
        distribution values at bin centers
    """

    if bins is None:
        dx = np.abs(np.diff(diffs))
        dx = dx[dx > 0]
        dx = np.mean(dx)
        bins = int((np.max(diffs) - np.min(diffs)) / dx)
        logger.debug('Using n_bins=%s to compute distribution', bins)
    counts, edges = np.histogram(diffs, bins=bins, range=range)
    centers = edges[:-1] + (np.diff(edges) / 2)
    if interpolate:
        indices = np.where(counts > 0)
        y = counts[indices]
        x = centers[indices]
        if len(x) > 1:
            interp = interp1d(x, y, bounds_error=False, fill_value=0)
            counts = interp(centers)
    counts = counts.astype(float) / counts.sum()
    return counts, centers
