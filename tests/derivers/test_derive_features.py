"""Run tests for feature derivations"""

import pytest

from sup3r.preprocessing import DataHandler

features = [
    'windspeed_100m',
    'winddirection_100m',
    'latitude_feature',
    'longitude_feature',
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
