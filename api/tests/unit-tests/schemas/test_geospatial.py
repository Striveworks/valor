import pytest
from pydantic import ValidationError

from velour_api import schemas


def test_GeoJSON():
    # test valid entries
    valid_point = schemas.metadata.geojson_from_dict(
        {"type": "Point", "coordinates": [125.2750725, 38.760525]}
    )
    assert type(valid_point.geometry()) is schemas.Point
    assert type(valid_point.model_dump()) is dict
    assert all(
        [
            key in ["type", "coordinates"]
            for key in valid_point.model_dump().keys()
        ]
    )
    assert valid_point.model_dump()["type"] == "Point"

    with pytest.raises(ValidationError):
        schemas.metadata.geojson_from_dict(
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

    valid_polygon = schemas.metadata.geojson_from_dict(
        {
            "type": "Polygon",
            "coordinates": [
                [
                    [125.2750725, 38.760525],
                    [125.3902365, 38.775069],
                    [125.5054005, 38.789613],
                    [125.5051935, 38.71402425],
                    [125.2750725, 38.760525],
                ],
            ],
        }
    )
    assert type(valid_polygon.geometry()) is schemas.Polygon
    assert type(valid_polygon.model_dump()) is dict
    assert all(
        [
            key in ["type", "coordinates"]
            for key in valid_polygon.model_dump().keys()
        ]
    )
    assert valid_polygon.model_dump()["type"] == "Polygon"

    with pytest.raises(ValidationError):
        schemas.metadata.geojson_from_dict(
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

    valid_multi = schemas.metadata.geojson_from_dict(
        {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [
                        [125.2750725, 38.760525],
                        [125.3902365, 38.775069],
                        [125.5054005, 38.789613],
                        [125.5051935, 38.71402425],
                        [125.2750725, 38.760525],
                    ],
                    [
                        [125.2750725, 38.760525],
                        [125.3902365, 38.775069],
                        [125.5054005, 38.789613],
                        [125.5051935, 38.71402425],
                        [125.2750725, 38.760525],
                    ],
                ],
                [
                    [
                        [125.2750725, 38.760525],
                        [125.3902365, 38.775069],
                        [125.5054005, 38.789613],
                        [125.5051935, 38.71402425],
                        [125.2750725, 38.760525],
                    ],
                    [
                        [125.2750725, 38.760525],
                        [125.3902365, 38.775069],
                        [125.5054005, 38.789613],
                        [125.5051935, 38.71402425],
                        [125.2750725, 38.760525],
                    ],
                ],
            ],
        }
    )
    assert type(valid_multi.geometry()) is schemas.MultiPolygon
    assert type(valid_multi.model_dump()) is dict
    assert all(
        [
            key in ["type", "coordinates"]
            for key in valid_multi.model_dump().keys()
        ]
    )
    assert valid_multi.model_dump()["type"] == "MultiPolygon"

    # invalids
    with pytest.raises(ValueError):
        schemas.metadata.geojson_from_dict(
            {
                "type": "fake_type",
                "coordinates": [
                    [
                        [125.2750725, 38.760525],
                    ]
                ],
            }
        )

    with pytest.raises(ValidationError):
        schemas.metadata.geojson_from_dict(
            {
                "type": "Polygon",
                "coordinates": "fake_string",
            }
        )

    with pytest.raises(ValidationError):
        schemas.metadata.geojson_from_dict(
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [125.2750725, 38.760525],
                    ]
                ],
            }
        ).geometry()

    with pytest.raises(ValidationError):
        schemas.metadata.geojson_from_dict(
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [125.2750725, 38.760525],
                        [125.3902365, 38.775069],
                    ]
                ],
            }
        ).geometry()
