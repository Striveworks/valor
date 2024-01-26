import pytest
import datetime

from velour import (
    Label,
    Annotation,
    Datum,
    GroundTruth,
    Prediction,
    Dataset,
    Model,
    Filter,
)
from velour.enums import AnnotationType, TaskType


def test_empty_filter():
    Filter()


def test_declarative_filtering():
    filters = [

        Dataset.name == "dataset1",
        Model.name == "model1",
        Datum.uid == "uid1",

        Label.key == "k1",
        Annotation.labels == Label(key="k2", value="v2"),

        Annotation.task_type.in_([TaskType.CLASSIFICATION, TaskType.DETECTION]),

        # geometry filters
        Annotation.raster.is_none(),
        Annotation.multipolygon.is_none(),
        Annotation.polygon.is_none(),
        Annotation.bounding_box.exists(),
        Annotation.bounding_box.area >= 1000,
        Annotation.bounding_box.area <= 5000,

        # metadata filters
        Dataset.metadata["arbitrary_numeric_key"] >= 10,
        Dataset.metadata["arbitrary_numeric_key"] < 20,
        
        Model.metadata["arbitrary_str_key"] == "arbitrary value",

        Datum.metadata["arbitrary_datetime_key"] >= datetime.timedelta(days=1),
        Datum.metadata["arbitrary_datetime_key"] <= datetime.timedelta(days=2),

        Annotation.metadata["myKey"] == "helloworld",

        # geospatial filters
    ]


