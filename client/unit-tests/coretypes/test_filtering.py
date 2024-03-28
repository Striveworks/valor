import datetime

import pytest

from valor import Annotation, Dataset, Filter, Label, Model
from valor.coretypes import _format_filter
from valor.schemas import Polygon
from valor.schemas.filters import Constraint


@pytest.fixture
def polygon() -> Polygon:
    coordinates = [
        [
            (125.2750725, 38.760525),
            (125.3902365, 38.775069),
            (125.5054005, 38.789613),
            (125.5051935, 38.71402425),
            (125.5049865, 38.6384355),
            (125.3902005, 38.6244225),
            (125.2754145, 38.6104095),
            (125.2752435, 38.68546725),
            (125.2750725, 38.760525),
        ]
    ]
    return Polygon(coordinates)


@pytest.fixture
def geojson(polygon: Polygon):
    return {"type": "Polygon", "coordinates": polygon.get_value()}


def test__format_filter(geojson, polygon):

    filter_object = Filter(
        dataset_names=["a", "b", "c"],
        model_names=["x", "y", "z"],
        label_scores=[Constraint(value=0.75, operator=">")],
        polygon_area=[
            Constraint(value=1000, operator=">"),
            Constraint(value=5000, operator="<"),
        ],
        raster_area=[
            Constraint(value=100, operator=">"),
            Constraint(value=500, operator="<"),
        ],
        dataset_metadata={
            "some_str": [Constraint(value="foobar", operator="==")],
            "some_float": [Constraint(value=0.123, operator=">=")],
            "some_datetime": [
                Constraint(
                    value={
                        "duration": datetime.timedelta(days=1).total_seconds()
                    },
                    operator=">",
                )
            ],
            "some_geospatial": [
                Constraint(
                    value=geojson,
                    operator="intersect",
                )
            ],
        },
    )

    filter_from_constraints = Filter.create(
        [
            Dataset.name.in_(["a", "b"]) | (Dataset.name == "c"),
            (Model.name == "x") | Model.name.in_(["y", "z"]),
            Label.score > 0.75,
            Annotation.polygon.area > 1000,
            Annotation.polygon.area < 5000,
            (Annotation.raster.area > 100) & (Annotation.raster.area < 500),
            Dataset.metadata["some_str"] == "foobar",
            Dataset.metadata["some_float"] >= 0.123,
            Dataset.metadata["some_datetime"] > datetime.timedelta(days=1),
            Dataset.metadata["some_geospatial"].intersects(polygon),
        ]
    )

    filter_from_dictionary = _format_filter(
        {
            "dataset_names": ["a", "b", "c"],
            "model_names": ["x", "y", "z"],
            "label_scores": [{"value": 0.75, "operator": ">"}],
            "polygon_area": [
                {"value": 1000, "operator": ">"},
                {"value": 5000, "operator": "<"},
            ],
            "raster_area": [
                {"value": 100, "operator": ">"},
                {"value": 500, "operator": "<"},
            ],
            "dataset_metadata": {
                "some_str": [{"value": "foobar", "operator": "=="}],
                "some_float": [{"value": 0.123, "operator": ">="}],
                "some_datetime": [
                    {
                        "value": {
                            "duration": datetime.timedelta(
                                days=1
                            ).total_seconds()
                        },
                        "operator": ">",
                    }
                ],
                "some_geospatial": [
                    {
                        "value": geojson,
                        "operator": "intersect",
                    }
                ],
            },
        }
    )

    assert filter_object == filter_from_constraints
    assert filter_object == filter_from_dictionary
