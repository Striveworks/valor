import pytest
from pydantic import ValidationError

from velour_api import schemas


def test_GeoJSON():
    # test valid entries
    valid_point = schemas.GeoJSON.from_dict(
        {"type": "Point", "coordinates": [125.2750725, 38.760525]}
    )
    assert type(valid_point.shape()) is schemas.Point
    assert type(valid_point.geometry.model_dump()) is dict
    assert all(
        [
            key in ["type", "coordinates"]
            for key in valid_point.geometry.model_dump().keys()
        ]
    )
    assert valid_point.geometry.model_dump()["type"] == "Point"

    valid_polygon = schemas.GeoJSON.from_dict(
        {
            "type": "Polygon",
            "coordinates": [
                [
                    [125.2750725, 38.760525],
                    [125.3902365, 38.775069],
                    [125.5054005, 38.789613],
                    [125.5051935, 38.71402425],
                ]
            ],
        }
    )
    assert type(valid_polygon.shape()) is schemas.Polygon
    assert type(valid_polygon.geometry.model_dump()) is dict
    assert all(
        [
            key in ["type", "coordinates"]
            for key in valid_polygon.geometry.model_dump().keys()
        ]
    )
    assert valid_polygon.geometry.model_dump()["type"] == "Polygon"

    valid_multi = schemas.GeoJSON.from_dict(
        {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [
                        [125.2750725, 38.760525],
                        [125.3902365, 38.775069],
                        [125.5054005, 38.789613],
                        [125.5051935, 38.71402425],
                    ],
                    [
                        [125.2750725, 38.760525],
                        [125.3902365, 38.775069],
                        [125.5054005, 38.789613],
                        [125.5051935, 38.71402425],
                    ],
                ],
                [
                    [
                        [125.2750725, 38.760525],
                        [125.3902365, 38.775069],
                        [125.5054005, 38.789613],
                        [125.5051935, 38.71402425],
                    ],
                    [
                        [125.2750725, 38.760525],
                        [125.3902365, 38.775069],
                        [125.5054005, 38.789613],
                        [125.5051935, 38.71402425],
                    ],
                ],
            ],
        }
    )
    assert type(valid_multi.shape()) is schemas.MultiPolygon
    assert type(valid_multi.geometry.model_dump()) is dict
    assert all(
        [
            key in ["type", "coordinates"]
            for key in valid_multi.geometry.model_dump().keys()
        ]
    )
    assert valid_multi.geometry.model_dump()["type"] == "MultiPolygon"

    # invalids
    with pytest.raises(ValueError):
        schemas.GeoJSON.from_dict(
            {
                "type": "fake_type",
                "coordinates": [
                    [
                        [125.2750725, 38.760525],
                    ]
                ],
            }
        ).shape()

    with pytest.raises(ValidationError):
        schemas.GeoJSON.from_dict(
            {
                "type": "Polygon",
                "coordinates": "fake_string",
            }
        ).shape()

    with pytest.raises(ValidationError):
        schemas.GeoJSON.from_dict(
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [125.2750725, 38.760525],
                    ]
                ],
            }
        ).shape()

    with pytest.raises(ValidationError):
        schemas.GeoJSON.from_dict(
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [125.2750725, 38.760525],
                        [125.3902365, 38.775069],
                    ]
                ],
            }
        ).shape()
