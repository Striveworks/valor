import numpy as np

from client.valor import (
    Annotation,
    Dataset,
    Datum,
    GroundTruth,
    Label,
    Model,
    Prediction,
    enums,
)
from client.valor.schemas import geometry
from client.valor.schemas.properties import (
    DictionaryProperty,
    GeometryProperty,
    LabelProperty,
    NumericProperty,
    StringProperty,
)


def test_label_typing():
    assert type(Label.key) is StringProperty
    assert type(Label.value) is StringProperty
    assert type(Label.score) is NumericProperty

    label = Label(key="k1", value="v1")
    assert type(label.key) is str
    assert type(label.value) is str
    assert label.score is None

    label = Label(key="k1", value="v1", score=None)
    assert type(label.key) is str
    assert type(label.value) is str
    assert label.score is None

    label = Label(key="k1", value="v1", score=1.4)
    assert type(label.key) is str
    assert type(label.value) is str
    assert type(label.score) is float


def test_annotation_typing():
    assert type(Annotation.task_type) is StringProperty
    assert type(Annotation.labels) is LabelProperty
    assert type(Annotation.metadata) is DictionaryProperty
    assert type(Annotation.bounding_box) is GeometryProperty
    assert type(Annotation.polygon) is GeometryProperty
    assert type(Annotation.raster) is GeometryProperty

    annotation = Annotation(
        task_type=enums.TaskType.CLASSIFICATION,
        labels=[],
    )
    assert type(annotation.task_type) is enums.TaskType
    assert type(annotation.labels) is list
    assert type(annotation.metadata) is dict
    assert annotation.bounding_box is None
    assert annotation.polygon is None
    assert annotation.raster is None

    bbox = geometry.BoundingBox.from_extrema(0, 1, 0, 1)
    polygon = geometry.Polygon(boundary=bbox.polygon)
    raster = geometry.Raster.from_numpy(np.zeros((10, 10)) == 0)
    annotation = Annotation(
        task_type=enums.TaskType.CLASSIFICATION,
        labels=[],
        metadata={},
        bounding_box=bbox,
        polygon=polygon,
        raster=raster,
    )
    assert type(annotation.task_type) is enums.TaskType
    assert type(annotation.labels) is list
    assert type(annotation.metadata) is dict
    assert type(annotation.bounding_box) is geometry.BoundingBox
    assert type(annotation.polygon) is geometry.Polygon
    assert type(annotation.raster) is geometry.Raster


def test_datum_typing():
    assert type(Datum.uid) is StringProperty
    assert type(Datum.metadata) is DictionaryProperty

    datum = Datum(uid="test")
    assert type(datum.uid) is str
    assert type(datum.metadata) is dict

    datum = Datum(uid="test", metadata={})
    assert type(datum.uid) is str
    assert type(datum.metadata) is dict


def test_groundtruth_typing():
    # GroundTruth doesn't use special properties.
    groundtruth = GroundTruth(datum=Datum(uid="uid"), annotations=[])
    assert type(groundtruth.datum) is Datum
    assert type(groundtruth.annotations) is list


def test_prediction_typing():
    # Prediction doesn't use special properties.
    prediction = Prediction(datum=Datum(uid="uid"), annotations=[])
    assert type(prediction.datum) is Datum
    assert type(prediction.annotations) is list


def test_dataset_typing():
    assert type(Dataset.name) is StringProperty
    assert type(Dataset.metadata) is DictionaryProperty

    dataset = Dataset(name="test")
    assert type(dataset.name) is str
    assert type(dataset.metadata) is dict

    dataset = Dataset(name="test", metadata={})
    assert type(dataset.name) is str
    assert type(dataset.metadata) is dict


def test_model_typing():
    assert type(Model.name) is StringProperty
    assert type(Model.metadata) is DictionaryProperty

    model = Model(name="test")
    assert type(model.name) is str
    assert type(model.metadata) is dict

    model = Model(name="test", metadata={})
    assert type(model.name) is str
    assert type(model.metadata) is dict
