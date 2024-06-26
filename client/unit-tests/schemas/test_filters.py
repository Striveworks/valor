import datetime

from valor import Annotation, Dataset, Datum, Filter, Label, Model
from valor.schemas import And


def test_empty_filter():
    assert Filter().to_dict() == {
        "datasets": None,
        "models": None,
        "datums": None,
        "annotations": None,
        "groundtruths": None,
        "predictions": None,
        "labels": None,
        "embeddings": None,
    }


def test_declarative_filtering():
    filters = Filter(
        datums=And(
            Datum.uid == "uid1",
            Datum.metadata["arbitrary_datetime_key"]
            >= datetime.timedelta(days=1),
            Datum.metadata["arbitrary_datetime_key"]
            <= datetime.timedelta(days=2),
        ),
        annotations=And(
            Dataset.name == "dataset1",
            Dataset.metadata["arbitrary_numeric_key"] >= 10,
            Dataset.metadata["arbitrary_numeric_key"] < 20,
            # geometry filters
            Annotation.raster.is_none(),
            Annotation.polygon.is_none(),
            Annotation.bounding_box.is_not_none(),
            Annotation.bounding_box.area >= 1000,
            Annotation.bounding_box.area <= 5000,
            Annotation.metadata["myKey"] == "helloworld",
            # label filters
            Label.key == "k2",
            Label.value == "v2",
        ),
        labels=And(
            Label.key == "k1",
            Label.score > 0.5,
            Label.score < 0.75,
        ),
        predictions=And(
            Model.name == "model1",
            Model.metadata["arbitrary_str_key"] == "arbitrary value",
        ),
    )

    assert filters.to_dict() == {
        "datasets": None,
        "models": None,
        "datums": {
            "op": "and",
            "args": [
                {
                    "lhs": {"name": "datum.uid", "key": None},
                    "rhs": {"type": "string", "value": "uid1"},
                    "op": "eq",
                },
                {
                    "lhs": {
                        "name": "datum.metadata",
                        "key": "arbitrary_datetime_key",
                    },
                    "rhs": {"type": "duration", "value": 86400.0},
                    "op": "gte",
                },
                {
                    "lhs": {
                        "name": "datum.metadata",
                        "key": "arbitrary_datetime_key",
                    },
                    "rhs": {"type": "duration", "value": 172800.0},
                    "op": "lte",
                },
            ],
        },
        "annotations": {
            "op": "and",
            "args": [
                {
                    "lhs": {"name": "dataset.name", "key": None},
                    "rhs": {"type": "string", "value": "dataset1"},
                    "op": "eq",
                },
                {
                    "lhs": {
                        "name": "dataset.metadata",
                        "key": "arbitrary_numeric_key",
                    },
                    "rhs": {"type": "integer", "value": 10},
                    "op": "gte",
                },
                {
                    "lhs": {
                        "name": "dataset.metadata",
                        "key": "arbitrary_numeric_key",
                    },
                    "rhs": {"type": "integer", "value": 20},
                    "op": "lt",
                },
                {
                    "lhs": {"name": "annotation.raster", "key": None},
                    "rhs": None,
                    "op": "isnull",
                },
                {
                    "lhs": {"name": "annotation.polygon", "key": None},
                    "rhs": None,
                    "op": "isnull",
                },
                {
                    "lhs": {"name": "annotation.bounding_box", "key": None},
                    "rhs": None,
                    "op": "isnotnull",
                },
                {
                    "lhs": {
                        "name": "annotation.bounding_box.area",
                        "key": None,
                    },
                    "rhs": {"type": "float", "value": 1000},
                    "op": "gte",
                },
                {
                    "lhs": {
                        "name": "annotation.bounding_box.area",
                        "key": None,
                    },
                    "rhs": {"type": "float", "value": 5000},
                    "op": "lte",
                },
                {
                    "lhs": {"name": "annotation.metadata", "key": "mykey"},
                    "rhs": {"type": "string", "value": "helloworld"},
                    "op": "eq",
                },
                {
                    "lhs": {"name": "label.key", "key": None},
                    "rhs": {"type": "string", "value": "k2"},
                    "op": "eq",
                },
                {
                    "lhs": {"name": "label.value", "key": None},
                    "rhs": {"type": "string", "value": "v2"},
                    "op": "eq",
                },
            ],
        },
        "groundtruths": None,
        "predictions": {
            "op": "and",
            "args": [
                {
                    "lhs": {"name": "model.name", "key": None},
                    "rhs": {"type": "string", "value": "model1"},
                    "op": "eq",
                },
                {
                    "lhs": {
                        "name": "model.metadata",
                        "key": "arbitrary_str_key",
                    },
                    "rhs": {"type": "string", "value": "arbitrary value"},
                    "op": "eq",
                },
            ],
        },
        "labels": {
            "op": "and",
            "args": [
                {
                    "lhs": {"name": "label.key", "key": None},
                    "rhs": {"type": "string", "value": "k1"},
                    "op": "eq",
                },
                {
                    "lhs": {"name": "label.score", "key": None},
                    "rhs": {"type": "float", "value": 0.5},
                    "op": "gt",
                },
                {
                    "lhs": {"name": "label.score", "key": None},
                    "rhs": {"type": "float", "value": 0.75},
                    "op": "lt",
                },
            ],
        },
        "embeddings": None,
    }
