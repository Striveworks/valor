import copy

import pytest

from velour import enums, schemas
from velour.schemas.core import _validate_href


def test__validate_href():
    _validate_href("http://test")
    _validate_href("https://test")

    with pytest.raises(ValueError) as e:
        _validate_href("test")
    assert "`href` must start with http:// or https://" in str(e)
    with pytest.raises(TypeError) as e:
        _validate_href(1)
    assert "passed something other than 'str'" in str(e)


def test_metadatum():
    schemas.Metadatum(key="test", value="test")
    schemas.Metadatum(key="test", value=1)
    schemas.Metadatum(key="test", value=1.0)
    # @TODO: Fix when geojson is implemented
    schemas.Metadatum(
        key="test", value=schemas.GeoJSON(type="test", coordinates=[])
    )

    with pytest.raises(TypeError) as e:
        schemas.Metadatum(key=123, value=123)
    assert "should always be of type string" in str(e)

    # Test supported value types
    with pytest.raises(NotImplementedError):
        schemas.Metadatum(key="test", value=(1, 2))
    assert "has unsupported type <class 'tuple'>"
    with pytest.raises(NotImplementedError):
        schemas.Metadatum(key="test", value=[1, 2])
    assert "has unsupported type <class 'list'>"

    # Test special type with name=href
    schemas.Metadatum(key="href", value="http://test")
    schemas.Metadatum(key="href", value="https://test")
    with pytest.raises(ValueError) as e:
        schemas.Metadatum(key="href", value="test")
    assert "`href` must start with http:// or https://" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Metadatum(key="href", value=1)
    assert "passed something other than 'str'" in str(e)

    # Check int to float conversion
    m = schemas.Metadatum(key="test", value=1)
    assert isinstance(m.value, float)


def test_dataset():
    # valid
    schemas.Dataset(
        name="test",
        metadata=[schemas.Metadatum(key="test", value=123)],
    )
    schemas.Dataset(
        name="test",
        id=None,
        metadata=[],
    )
    schemas.Dataset(
        name="test",
        id=1,
        metadata=[],
    )
    schemas.Dataset(
        name="test",
    )

    # test `__post_init__`
    with pytest.raises(TypeError):
        schemas.Dataset(name=123)
    with pytest.raises(TypeError):
        schemas.Dataset(name="123", id="123")
    with pytest.raises(TypeError):
        schemas.Dataset(name="123", metadata=1)
    with pytest.raises(TypeError):
        schemas.Dataset(name="123", metadata=[1])


def test_model():
    # valid
    schemas.Model(
        name="test",
        metadata=[schemas.Metadatum(key="test", value=123)],
    )
    schemas.Model(
        name="test",
        id=None,
        metadata=[],
    )
    schemas.Model(
        name="test",
        id=1,
        metadata=[],
    )
    schemas.Model(
        name="test",
    )

    # test `__post_init__`
    with pytest.raises(TypeError):
        schemas.Model(name=123)
    with pytest.raises(TypeError):
        schemas.Model(name="123", id="123")
    with pytest.raises(TypeError):
        schemas.Model(name="123", metadata=1)
    with pytest.raises(TypeError):
        schemas.Model(name="123", metadata=[1])


def test_datum():
    schemas.Datum(uid="123")
    schemas.Datum(uid="123", metadata=[])
    schemas.Datum(uid="123", metadata=[schemas.Metadatum(key="name", value=1)])
    schemas.Datum(uid="123", dataset="dataset")

    # test `__post_init__`
    with pytest.raises(TypeError):
        schemas.Datum(uid=123)
    with pytest.raises(TypeError):
        schemas.Datum(uid="123", metadata=1)
    with pytest.raises(TypeError):
        schemas.Datum(uid="123", metadata=[1])
    with pytest.raises(TypeError):
        schemas.Datum(uid="123", dataset=None)


def test_annotation(bbox, polygon, raster, metadatum):
    # valid
    schemas.Annotation(task_type=enums.TaskType.DETECTION, bounding_box=bbox)
    schemas.Annotation(task_type=enums.TaskType.DETECTION, polygon=polygon)
    schemas.Annotation(task_type=enums.TaskType.DETECTION, raster=raster)
    schemas.Annotation(task_type=enums.TaskType.SEGMENTATION, raster=raster)
    schemas.Annotation(
        task_type=enums.TaskType.DETECTION,
        bounding_box=bbox,
        polygon=polygon,
        raster=raster,
    )
    schemas.Annotation(task_type=enums.TaskType.CLASSIFICATION)
    schemas.Annotation(task_type=enums.TaskType.CLASSIFICATION, metadata=[])
    schemas.Annotation(
        task_type=enums.TaskType.CLASSIFICATION, metadata=[metadatum]
    )

    # test `__post_init__`
    with pytest.raises(ValueError) as e:
        schemas.Annotation(task_type="something")
    assert "is not a valid TaskType" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(
            task_type=enums.TaskType.DETECTION, bounding_box=polygon
        )
    assert "should be of type `velour.schemas.BoundingBox" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(task_type=enums.TaskType.DETECTION, polygon=bbox)
    assert "should be of type `velour.schemas.Polygon`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(
            task_type=enums.TaskType.DETECTION, multipolygon=bbox
        )
    assert "should be of type `velour.schemas.MultiPolygon`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(task_type=enums.TaskType.DETECTION, raster=bbox)
    assert "should be of type `velour.schemas.Raster`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(
            task_type=enums.TaskType.CLASSIFICATION, metadata=metadatum
        )
    assert "should be of type `list`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(
            task_type=enums.TaskType.CLASSIFICATION, metadata=[123]
        )
    assert "should be of type `velour.schemas.Metadatum`" in str(e)


def test_label():
    # valid
    l1 = schemas.Label(key="test", value="value")

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        schemas.Label(key=123, value="123")
    assert "should be of type `str`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Label(key="123", value=123)
    assert "should be of type `str`" in str(e)

    # test member fn `tuple`
    assert l1.tuple() == ("test", "value", None)

    # test member fn `__eq__`
    l2 = schemas.Label(key="test", value="value")
    assert l1 == l2

    # test member fn `__hash__`
    assert l1.__hash__() == l2.__hash__()


def test_scored_label():
    l1 = schemas.Label(key="test", value="value")

    # valid
    s1 = schemas.Label(key="test", value="value", score=0.5)
    s2 = schemas.Label(key="test", value="value", score=0.5)
    s3 = schemas.Label(key="test", value="value", score=0.1)
    s4 = schemas.Label(key="test", value="other", score=0.5)
    s5 = schemas.Label(key="other", value="value", score=0.5)

    # test `__post_init__`

    with pytest.raises(TypeError) as e:
        schemas.Label(key="k", value="v", score="0.5")
    assert "score should be of type `float`" in str(e)

    # test property `key`
    assert l1.key == "test"

    # test property `value`
    assert l1.value == "value"

    # test member fn `__eq__`
    assert s1 == s2
    assert not s1 == s3
    assert not s1 == s4
    assert not s1 == s5
    assert not s1 == 123
    assert not s1 == "123"

    # test member fn `__eq__`
    assert not s1 != s2
    assert s1 != s3
    assert s1 != s4
    assert s1 != s5
    assert s1 != 123
    assert s1 != "123"

    # test member fn `__hash__`
    assert s1.__hash__() == s2.__hash__()
    assert s1.__hash__() != s3.__hash__()
    assert s1.__hash__() != s4.__hash__()
    assert s1.__hash__() != s5.__hash__()


def test_groundtruth_annotation():
    l1 = schemas.Label(key="test", value="value")
    l2 = schemas.Label(key="test", value="other")
    l3 = schemas.Label(key="other", value="value")

    # valid
    schemas.Annotation(
        task_type=enums.TaskType.CLASSIFICATION,
        labels=[l1, l2, l3],
    )

    # test `__post_init__`
    with pytest.raises(ValueError) as e:
        schemas.Annotation(task_type="soemthing", labels=[l1])
    assert "is not a valid TaskType" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(task_type=enums.TaskType.CLASSIFICATION, labels=l1)
    assert "should be of type `list`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(
            task_type=enums.TaskType.CLASSIFICATION, labels=[l1, l2, "label"]
        )
    assert (
        "elements of labels should be of type `velour.schemas.Label`" in str(e)
    )


def test_prediction_annotation():
    l1 = schemas.Label(key="test", value="value")
    l2 = schemas.Label(key="test", value="other")
    l3 = schemas.Label(key="other", value="value")

    s1 = copy.deepcopy(l1)
    s1.score = 0.5
    s2 = copy.deepcopy(l2)
    s2.score = 0.5
    s3 = copy.deepcopy(l3)
    s3.score = 1.0

    # valid
    schemas.Annotation(
        task_type=enums.TaskType.CLASSIFICATION, labels=[s1, s2, s3]
    )

    # test `__post_init__`
    with pytest.raises(ValueError) as e:
        schemas.Annotation(task_type="something", labels=[s1, s2, s3])
    assert "is not a valid TaskType" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(task_type=enums.TaskType.CLASSIFICATION, labels=s1)
    assert "should be of type `list`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(
            task_type=enums.TaskType.CLASSIFICATION, labels=[s1, s2, "label"]
        )
    assert (
        "elements of labels should be of type `velour.schemas.Label`" in str(e)
    )


def test_groundtruth():
    label = schemas.Label(key="test", value="value")
    datum = schemas.Datum(uid="somefile")
    gts = [
        schemas.Annotation(
            task_type=enums.TaskType.CLASSIFICATION, labels=[label]
        ),
        schemas.Annotation(
            task_type=enums.TaskType.CLASSIFICATION, labels=[label]
        ),
    ]

    # valid
    schemas.GroundTruth(
        datum=datum,
        annotations=gts,
    )

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        schemas.GroundTruth(
            datum="datum",
            annotations=gts,
        )
    assert "datum should be of type `velour.schemas.Datum`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.GroundTruth(
            datum=datum,
            annotations=gts[0],
        )
    assert "annotations should be of type `list`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.GroundTruth(
            datum=datum,
            annotations=[gts[0], gts[1], "annotation"],
        )
    assert "should be of type `velour.schemas.Annotation`" in str(e)


def test_prediction():
    scored_label = schemas.Label(key="test", value="value", score=1.0)
    datum = schemas.Datum(uid="somefile")
    pds = [
        schemas.Annotation(
            task_type=enums.TaskType.CLASSIFICATION,
            labels=[scored_label],
        ),
        schemas.Annotation(
            task_type=enums.TaskType.CLASSIFICATION,
            labels=[scored_label],
        ),
    ]

    # valid
    schemas.Prediction(datum=datum, annotations=pds)
    schemas.Prediction(datum=datum, annotations=pds, model="test")

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        schemas.Prediction(datum="datum", annotations=pds)
    assert "should be of type `velour.schemas.Datum`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Prediction(
            datum=datum,
            annotations=pds[0],
        )
    assert "should be of type `list`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Prediction(
            datum=datum,
            annotations=[pds[0], pds[1], "annotation"],
        )
    assert "should be of type `velour.schemas.Annotation`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Prediction(
            datum=datum,
            annotations=pds,
            model=1234,
        )
    assert "should be of type `str`" in str(e)

    with pytest.raises(ValueError) as e:
        schemas.Prediction(
            datum=datum,
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(key="test", value="value", score=0.8),
                        schemas.Label(key="test", value="other", score=0.1),
                    ],
                )
            ],
            model="",
        )
    assert "for label key test got scores summing to 0.9" in str(e)
