import numpy as np

from valor import (
    Annotation,
    Dataset,
    Datum,
    GroundTruth,
    Label,
    Model,
    Prediction,
    enums,
)
from valor.schemas import (
    Box,
    Context,
    Dictionary,
    Float,
    List,
    Polygon,
    Raster,
    String,
    TaskTypeEnum,
    Text,
)


def test_label_typing():
    assert type(Label.key) is String
    assert type(Label.value) is String
    assert type(Label.score) is Float

    label = Label(key="k1", value="v1")
    assert type(label.key) is str
    assert type(label.value) is str
    assert label.score is None

    label = Label(key="k1", value="v1", score=None)
    assert type(label.key) is str
    assert type(label.value) is str
    assert label.score is None

    label = Label(key="k1", value="v1", score=1.0)
    assert type(label.key) is str
    assert type(label.value) is str
    assert type(label.score) is float


def test_annotation_typing():
    assert type(Annotation.task_type) is TaskTypeEnum
    assert type(Annotation.labels) is List[Label]
    assert type(Annotation.metadata) is Dictionary
    assert type(Annotation.bounding_box) is Box
    assert type(Annotation.polygon) is Polygon
    assert type(Annotation.raster) is Raster
    assert type(Annotation.text) is Text
    assert type(Annotation.context) is Context

    annotation = Annotation(
        task_type=enums.TaskType.CLASSIFICATION,
        labels=[],
    )
    assert type(annotation.task_type) is enums.TaskType
    assert type(annotation.labels) is List[Label]
    assert type(annotation.metadata) is Dictionary
    assert annotation.bounding_box is None
    assert annotation.polygon is None
    assert annotation.raster is None
    assert annotation.text is None
    assert annotation.context is None

    bbox = Box.from_extrema(0, 1, 0, 1)
    polygon = Polygon([bbox.boundary])
    raster = Raster.from_numpy(np.zeros((10, 10)) == 0)
    annotation = Annotation(
        task_type=enums.TaskType.CLASSIFICATION,
        labels=[],
        metadata={},
        bounding_box=bbox,
        polygon=polygon,
        raster=raster,
    )
    assert type(annotation.task_type) is enums.TaskType
    assert type(annotation.labels) is List[Label]
    assert type(annotation.metadata) is Dictionary
    assert type(annotation.bounding_box) is Box
    assert type(annotation.polygon) is Polygon
    assert type(annotation.raster) is Raster
    assert annotation.text is None
    assert annotation.context is None

    text = "Example text."
    context = ["context 1", "context 2"]
    annotation = Annotation(
        task_type=enums.TaskType.TEXT_GENERATION,
        metadata={},
        text=text,
        context=context,
    )

    assert type(annotation.task_type) is enums.TaskType
    assert type(annotation.labels) is List[Label]
    assert type(annotation.metadata) is Dictionary
    assert annotation.bounding_box is None
    assert annotation.polygon is None
    assert annotation.raster is None
    assert type(annotation.text) is str
    assert type(annotation.context) is Context


def test_datum_typing():
    assert type(Datum.uid) is String
    assert type(Datum.metadata) is Dictionary
    assert type(Datum.text) is Text

    datum = Datum(uid="test")
    assert type(datum.uid) is str
    assert type(datum.metadata) is Dictionary
    assert datum.text is None

    text = "Example text."
    datum = Datum(uid="test", text=text, metadata={})
    assert type(datum.uid) is str
    assert type(datum.metadata) is Dictionary
    assert type(datum.text) is str


def test_groundtruth_typing():
    # GroundTruth doesn't use special properties.
    groundtruth = GroundTruth(datum=Datum(uid="uid"), annotations=[])
    assert type(groundtruth.datum) is Datum
    assert type(groundtruth.annotations) is List[Annotation]


def test_prediction_typing():
    # Prediction doesn't use special properties.
    prediction = Prediction(datum=Datum(uid="uid"), annotations=[])
    assert type(prediction.datum) is Datum
    assert type(prediction.annotations) is List[Annotation]


def test_dataset_typing():
    assert type(Dataset.name) is String
    assert type(Dataset.metadata) is Dictionary

    dataset = Dataset(name="test")
    assert type(dataset.name) is str
    assert type(dataset.metadata) is Dictionary

    dataset = Dataset(name="test", metadata={})
    assert type(dataset.name) is str
    assert type(dataset.metadata) is Dictionary


def test_model_typing():
    assert type(Model.name) is String
    assert type(Model.metadata) is Dictionary

    model = Model(name="test")
    assert type(model.name) is str
    assert type(model.metadata) is Dictionary

    model = Model(name="test", metadata={})
    assert type(model.name) is str
    assert type(model.metadata) is Dictionary
