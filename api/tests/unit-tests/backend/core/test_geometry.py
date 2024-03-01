import pytest

from valor_api.backend.core.geometry import (
    _convert_multipolygon_to_box,
    _convert_multipolygon_to_polygon,
    _convert_raster_to_multipolygon,
)


def test_multipolygon_not_implemented():
    for f in [
        _convert_multipolygon_to_box,
        _convert_multipolygon_to_polygon,
        _convert_raster_to_multipolygon,
    ]:
        with pytest.raises(NotImplementedError):
            f([])
