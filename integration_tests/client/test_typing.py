import numpy as np

from velour import (
    Label,
    Annotation,
    Datum,
    GroundTruth,
    Prediction,
    Dataset,
    Model,
)
from velour.schemas.properties import (
    NumericProperty,
    StringProperty,
    LabelProperty,
    GeometryProperty,
    GeospatialProperty,
    DictionaryProperty,
)
from velour import enums
from velour.schemas import geometry

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
    assert type(Annotation.multipolygon) is GeometryProperty
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
    assert annotation.multipolygon is None
    assert annotation.raster is None

    bbox = geometry.BoundingBox.from_extrema(0,1,0,1)
    polygon = geometry.Polygon(boundary=bbox.polygon)
    multipolygon = geometry.MultiPolygon(polygons=[polygon])
    raster = geometry.Raster.from_numpy(np.zeros((10,10)) == 0)
    annotation = Annotation(
        task_type=enums.TaskType.CLASSIFICATION,
        labels=[],
        metadata={},
        bounding_box=bbox,
        polygon=polygon,
        multipolygon=multipolygon,
        raster=raster,
    )
    assert type(annotation.task_type) is enums.TaskType
    assert type(annotation.labels) is list
    assert type(annotation.metadata) is dict
    assert type(annotation.bounding_box) is geometry.BoundingBox
    assert type(annotation.polygon) is geometry.Polygon
    assert type(annotation.multipolygon) is geometry.MultiPolygon
    assert type(annotation.raster) is geometry.Raster


def test_datum_typing():
    assert type(Datum.uid) is StringProperty
    assert type(Datum.metadata) is DictionaryProperty
    assert type(Datum.geospatial) is GeospatialProperty

    datum = Datum(uid="test")
    assert type(datum.uid) is str
    assert type(datum.metadata) is dict
    assert datum.geospatial is None

    datum = Datum(uid="test", metadata={}, geospatial={})
    assert type(datum.uid) is str
    assert type(datum.metadata) is dict
    assert type(datum.geospatial) is dict


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
    assert type(Dataset.geospatial) is GeospatialProperty

    dataset = Dataset(name="test")
    assert type(dataset.name) is str
    assert type(dataset.metadata) is dict
    assert dataset.geospatial is None

    dataset = Dataset(name="test", metadata={}, geospatial={})
    assert type(dataset.name) is str
    assert type(dataset.metadata) is dict
    assert type(dataset.geospatial) is dict


def test_model_typing():
    assert type(Model.name) is StringProperty
    assert type(Model.metadata) is DictionaryProperty
    assert type(Model.geospatial) is GeospatialProperty

    model = Model(name="test")
    assert type(model.name) is str
    assert type(model.metadata) is dict
    assert model.geospatial is None

    model = Model(name="test", metadata={}, geospatial={})
    assert type(model.name) is str
    assert type(model.metadata) is dict
    assert type(model.geospatial) is dict
