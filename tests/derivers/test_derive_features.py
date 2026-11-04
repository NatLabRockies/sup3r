"""Run tests for feature derivations"""

import numpy as np

import pytest

from sup3r.preprocessing import DataHandler

features = [
    'windspeed_100m',
    'winddirection_100m',
    'latitude_feature',
    'longitude_feature',
    'relative_time_feature',
    'soy_encoding',
    'sod_encoding',
    'lat_encoding',
    'lon_encoding',
]


@pytest.mark.parametrize('feature', features)
def test_derive_feature(feature):
    """Make sure features can be derived correctly"""

    handler = DataHandler(
        pytest.FP_WTK,
        features=[feature],
        time_slice=slice(0, 5),
        shape=(20, 20),
        target=(39.01, -105.15),
    )
    assert handler.data.shape == (20, 20, 5, 1)


def test_relative_time_feature():
    """Relative time starts at zero and preserves source time increments."""
    handler = DataHandler(
        pytest.FP_WTK,
        features=['relative_time_feature'],
        time_slice=slice(0, 5),
        shape=(20, 20),
        target=(39.01, -105.15),
    )

    relative_time = handler['relative_time_feature'].values[0, 0]
    expected_steps = np.diff(handler.time_index.values).astype(
        'timedelta64[s]'
    ).astype(np.int64)

    assert relative_time.dtype == np.float32
    assert relative_time[0] == 0
    np.testing.assert_array_equal(np.diff(relative_time), expected_steps)
