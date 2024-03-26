import datetime

import pytest

from valor import Annotation, Constraint, Dataset, Filter, Label, Model
from valor.coretypes import _format_filter


@pytest.fixture
def geojson() -> dict:
    coordinates = [
        [
            [125.2750725, 38.760525],
            [125.3902365, 38.775069],
            [125.5054005, 38.789613],
            [125.5051935, 38.71402425],
            [125.5049865, 38.6384355],
            [125.3902005, 38.6244225],
            [125.2754145, 38.6104095],
            [125.2752435, 38.68546725],
            [125.2750725, 38.760525],
        ]
    ]
    return {"type": "Polygon", "coordinates": coordinates}


def test__format_filter(geojson):

    filter_object = Filter(
        dataset_names=["a", "b", "c"],
        model_names=["x", "y", "z"],
        label_scores=[Constraint(value=0.75, operator=">")],
        polygon_area=[
            Constraint(value=1000, operator=">"),
            Constraint(value=5000, operator="<"),
        ],
        dataset_metadata={
            "some_str": [Constraint(value="foobar", operator="==")],
            "some_float": [Constraint(value=0.123, operator=">=")],
            "some_datetime": [
                Constraint(
                    value={
                        "duration": str(
                            datetime.timedelta(days=1).total_seconds()
                        )
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

    filter_from_constraints = _format_filter(
        [
            Dataset.name.in_(["a", "b", "c"]),  # type: ignore - filter type error
            Model.name.in_(["x", "y", "z"]),  # type: ignore - filter type error
            Label.score > 0.75,  # type: ignore - filter type error
            Annotation.polygon.area > 1000,  # type: ignore - filter type error
            Annotation.polygon.area < 5000,  # type: ignore - filter type error
            Dataset.metadata["some_str"] == "foobar",  # type: ignore - filter type error
            Dataset.metadata["some_float"] >= 0.123,  # type: ignore - filter type error
            Dataset.metadata["some_datetime"] > datetime.timedelta(days=1),  # type: ignore - filter type error
            Dataset.metadata["some_geospatial"].intersect(geojson),  # type: ignore - filter type error
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
            "dataset_metadata": {
                "some_str": [{"value": "foobar", "operator": "=="}],
                "some_float": [{"value": 0.123, "operator": ">="}],
                "some_datetime": [
                    {
                        "value": {
                            "duration": str(
                                datetime.timedelta(days=1).total_seconds()
                            )
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
