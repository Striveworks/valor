import datetime
from typing import Dict, List, Tuple, Union

import pytest

from valor import Annotation, Dataset, Filter, Label, Model
from valor.schemas import And, Polygon


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
def geojson(
    polygon: Polygon,
) -> Dict[str, Union[str, List[List[Tuple[float, float]]]]]:
    return {"type": "Polygon", "coordinates": polygon.get_value()}


def test_complex_filter(
    geojson: Dict[str, Union[str, List[List[Tuple[float, float]]]]],
    polygon: Polygon,
):
    filter_from_constraints = Filter(
        annotations=And(
            Dataset.name.in_(["a", "b", "c"]),
            (Model.name == "x") | Model.name.in_(["y", "z"]),
            Label.score > 0.75,
            Annotation.polygon.area > 1000,
            Annotation.polygon.area < 5000,
            (Annotation.raster.area > 100) & (Annotation.raster.area < 500),
            Dataset.metadata["some_str"] == "foobar",
            Dataset.metadata["some_float"] >= 0.123,
            Dataset.metadata["some_datetime"] > datetime.timedelta(days=1),
            Dataset.metadata["some_geospatial"].intersects(polygon),  # type: ignore - issue #605
        )
    )

    assert filter_from_constraints.to_dict() == {
        "datasets": None,
        "models": None,
        "datums": None,
        "annotations": {
            "op": "and",
            "args": [
                {
                    "op": "or",
                    "args": [
                        {
                            "lhs": {"name": "dataset.name", "key": None},
                            "rhs": {"type": "string", "value": "a"},
                            "op": "eq",
                        },
                        {
                            "lhs": {"name": "dataset.name", "key": None},
                            "rhs": {"type": "string", "value": "b"},
                            "op": "eq",
                        },
                        {
                            "lhs": {"name": "dataset.name", "key": None},
                            "rhs": {"type": "string", "value": "c"},
                            "op": "eq",
                        },
                    ],
                },
                {
                    "op": "or",
                    "args": [
                        {
                            "lhs": {"name": "model.name", "key": None},
                            "rhs": {"type": "string", "value": "x"},
                            "op": "eq",
                        },
                        {
                            "lhs": {"name": "model.name", "key": None},
                            "rhs": {"type": "string", "value": "y"},
                            "op": "eq",
                        },
                        {
                            "lhs": {"name": "model.name", "key": None},
                            "rhs": {"type": "string", "value": "z"},
                            "op": "eq",
                        },
                    ],
                },
                {
                    "lhs": {"name": "label.score", "key": None},
                    "rhs": {"type": "float", "value": 0.75},
                    "op": "gt",
                },
                {
                    "lhs": {"name": "annotation.polygon.area", "key": None},
                    "rhs": {"type": "float", "value": 1000},
                    "op": "gt",
                },
                {
                    "lhs": {"name": "annotation.polygon.area", "key": None},
                    "rhs": {"type": "float", "value": 5000},
                    "op": "lt",
                },
                {
                    "lhs": {"name": "annotation.raster.area", "key": None},
                    "rhs": {"type": "float", "value": 100},
                    "op": "gt",
                },
                {
                    "lhs": {"name": "annotation.raster.area", "key": None},
                    "rhs": {"type": "float", "value": 500},
                    "op": "lt",
                },
                {
                    "lhs": {"name": "dataset.metadata", "key": "some_str"},
                    "rhs": {"type": "string", "value": "foobar"},
                    "op": "eq",
                },
                {
                    "lhs": {"name": "dataset.metadata", "key": "some_float"},
                    "rhs": {"type": "float", "value": 0.123},
                    "op": "gte",
                },
                {
                    "lhs": {
                        "name": "dataset.metadata",
                        "key": "some_datetime",
                    },
                    "rhs": {"type": "duration", "value": 86400.0},
                    "op": "gt",
                },
                {
                    "lhs": {
                        "name": "dataset.metadata",
                        "key": "some_geospatial",
                    },
                    "rhs": {
                        "type": "polygon",
                        "value": [
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
                        ],
                    },
                    "op": "intersects",
                },
            ],
        },
        "groundtruths": None,
        "predictions": None,
        "labels": None,
        "embeddings": None,
    }
