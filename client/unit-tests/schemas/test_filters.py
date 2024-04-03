import datetime
from dataclasses import asdict

from valor import Annotation, Dataset, Datum, Filter, Label, Model
from valor.enums import TaskType


def test_empty_filter():
    f = asdict(Filter())
    assert f == {
        "dataset_names": None,
        "dataset_metadata": None,
        "model_names": None,
        "model_metadata": None,
        "datum_uids": None,
        "datum_metadata": None,
        "task_types": None,
        "annotation_metadata": None,
        "require_bounding_box": None,
        "box_area": None,
        "require_polygon": None,
        "polygon_area": None,
        "require_raster": None,
        "raster_area": None,
        "labels": None,
        "label_ids": None,
        "label_keys": None,
        "label_scores": None,
    }


def test_declarative_filtering():
    filters = [
        Dataset.name == "dataset1",
        Model.name == "model1",
        Datum.uid == "uid1",
        Label.key == "k1",
        Label.score > 0.5,
        Label.score < 0.75,
        Annotation.labels == [Label(key="k2", value="v2")],
        Annotation.task_type.in_(
            [TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION]
        ),
        # geometry filters
        Annotation.raster.is_none(),
        Annotation.polygon.is_none(),
        Annotation.box.is_not_none(),
        Annotation.box.area >= 1000,
        Annotation.box.area <= 5000,
        # metadata filters
        Dataset.metadata["arbitrary_numeric_key"] >= 10,
        Dataset.metadata["arbitrary_numeric_key"] < 20,
        Model.metadata["arbitrary_str_key"] == "arbitrary value",
        Datum.metadata["arbitrary_datetime_key"] >= datetime.timedelta(days=1),
        Datum.metadata["arbitrary_datetime_key"] <= datetime.timedelta(days=2),
        Annotation.metadata["myKey"] == "helloworld",
    ]

    f = asdict(Filter.create(filters))
    assert f == {
        "dataset_names": ["dataset1"],
        "dataset_metadata": {
            "arbitrary_numeric_key": [
                {"value": 10, "operator": ">="},
                {"value": 20, "operator": "<"},
            ]
        },
        "model_names": ["model1"],
        "model_metadata": {
            "arbitrary_str_key": [
                {"value": "arbitrary value", "operator": "=="}
            ]
        },
        "datum_uids": ["uid1"],
        "datum_metadata": {
            "arbitrary_datetime_key": [
                {"value": {"duration": 86400.0}, "operator": ">="},
                {"value": {"duration": 172800.0}, "operator": "<="},
            ]
        },
        "task_types": [
            TaskType.CLASSIFICATION.value,
            TaskType.OBJECT_DETECTION.value,
        ],
        "require_bounding_box": True,
        "box_area": [
            {"value": 1000, "operator": ">="},
            {"value": 5000, "operator": "<="},
        ],
        "annotation_metadata": {
            "mykey": [{"value": "helloworld", "operator": "=="}]
        },
        "require_polygon": False,
        "polygon_area": None,
        "require_raster": False,
        "raster_area": None,
        "labels": [{"k2": "v2"}],
        "label_ids": None,
        "label_keys": ["k1"],
        "label_scores": [
            {"value": 0.5, "operator": ">"},
            {"value": 0.75, "operator": "<"},
        ],
    }
