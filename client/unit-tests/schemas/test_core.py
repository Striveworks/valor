import copy

import pytest

from velour import enums
from velour.coretypes import Annotation, Datum, GroundTruth, Label, Prediction
from velour.exceptions import SchemaTypeError


def test_datum():
    Datum(uid="123")
    Datum(uid="123", metadata={})
    Datum(uid="123", metadata={"name": 1})

    # test `__post_init__`
    with pytest.raises(SchemaTypeError):
        Datum(uid=123)
    with pytest.raises(SchemaTypeError):
        Datum(uid="123", metadata=1)
    with pytest.raises(SchemaTypeError):
        Datum(uid="123", metadata=[1])


def test_annotation(bbox, polygon, raster, labels, metadata):
    # valid
    Annotation(
        task_type=enums.TaskType.DETECTION, bounding_box=bbox, labels=labels
    )
    Annotation(
        task_type=enums.TaskType.DETECTION, polygon=polygon, labels=labels
    )
    Annotation(
        task_type=enums.TaskType.DETECTION, raster=raster, labels=labels
    )
    Annotation(
        task_type=enums.TaskType.SEGMENTATION, raster=raster, labels=labels
    )
    Annotation(
        task_type=enums.TaskType.DETECTION,
        labels=labels,
        bounding_box=bbox,
        polygon=polygon,
        raster=raster,
    )
    Annotation(task_type=enums.TaskType.CLASSIFICATION, labels=labels)
    Annotation(
        task_type=enums.TaskType.CLASSIFICATION, labels=labels, metadata={}
    )
    Annotation(
        task_type=enums.TaskType.CLASSIFICATION,
        labels=labels,
        metadata=metadata,
    )

    # test `__post_init__`
    with pytest.raises(ValueError) as e:
        Annotation(task_type="something", labels=labels)
    assert "is not a valid TaskType" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        Annotation(
            task_type=enums.TaskType.DETECTION,
            labels=labels,
            bounding_box=polygon,
        )
    assert "`bounding_box` should be of type" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        Annotation(
            task_type=enums.TaskType.DETECTION,
            labels=labels,
            polygon=bbox,
        )
    assert "`polygon` should be of type" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        Annotation(
            task_type=enums.TaskType.DETECTION,
            labels=labels,
            multipolygon=bbox,
        )
    assert "`multipolygon` should be of type" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        Annotation(
            task_type=enums.TaskType.DETECTION,
            labels=labels,
            raster=bbox,
        )
    assert "`raster` should be of type" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        Annotation(
            task_type=enums.TaskType.CLASSIFICATION,
            labels=labels,
            metadata=[1234],
        )
    assert "`metadata` should be of type" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        Annotation(
            task_type=enums.TaskType.CLASSIFICATION,
            labels=labels,
            metadata={1: 1},
        )
    assert "`metadatum key` should be of type" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        Annotation(
            task_type=enums.TaskType.CLASSIFICATION,
            labels=labels,
            metadata={"test": None},
        )
    assert "`metadatum value` should be of type" in str(e)


def test_groundtruth_annotation():
    l1 = Label(key="test", value="value")
    l2 = Label(key="test", value="other")
    l3 = Label(key="other", value="value")

    # valid
    Annotation(
        task_type=enums.TaskType.CLASSIFICATION,
        labels=[l1, l2, l3],
    )

    # test `__post_init__`
    with pytest.raises(ValueError) as e:
        Annotation(task_type="soemthing", labels=[l1])
    assert "is not a valid TaskType" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        Annotation(task_type=enums.TaskType.CLASSIFICATION, labels=l1)
    assert "`labels` should be of type" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        Annotation(
            task_type=enums.TaskType.CLASSIFICATION, labels=[l1, l2, "label"]
        )
    assert "`label` should be of type" in str(e)


def test_prediction_annotation():
    l1 = Label(key="test", value="value")
    l2 = Label(key="test", value="other")
    l3 = Label(key="other", value="value")

    s1 = copy.deepcopy(l1)
    s1.score = 0.5
    s2 = copy.deepcopy(l2)
    s2.score = 0.5
    s3 = copy.deepcopy(l3)
    s3.score = 1.0

    # valid
    Annotation(task_type=enums.TaskType.CLASSIFICATION, labels=[s1, s2, s3])

    # test `__post_init__`
    with pytest.raises(ValueError) as e:
        Annotation(task_type="something", labels=[s1, s2, s3])
    assert "is not a valid TaskType" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        Annotation(task_type=enums.TaskType.CLASSIFICATION, labels=s1)
    assert "`labels` should be of type" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        Annotation(
            task_type=enums.TaskType.CLASSIFICATION, labels=[s1, s2, "label"]
        )
    assert "`label` should be of type" in str(e)


def test_groundtruth():
    label = Label(key="test", value="value")
    datum = Datum(uid="somefile")
    gts = [
        Annotation(task_type=enums.TaskType.CLASSIFICATION, labels=[label]),
        Annotation(task_type=enums.TaskType.CLASSIFICATION, labels=[label]),
    ]

    # valid
    GroundTruth(
        datum=datum,
        annotations=gts,
    )

    # test `__post_init__`
    with pytest.raises(SchemaTypeError) as e:
        GroundTruth(
            datum="datum",
            annotations=gts,
        )
    assert "`datum` should be of type" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        GroundTruth(
            datum=datum,
            annotations=gts[0],
        )
    assert "`annotations` should be of type" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        GroundTruth(
            datum=datum,
            annotations=[gts[0], gts[1], "annotation"],
        )
    assert "`annotation` should be of type" in str(e)


def test_prediction():
    scored_label = Label(key="test", value="value", score=1.0)
    datum = Datum(uid="somefile")
    pds = [
        Annotation(
            task_type=enums.TaskType.CLASSIFICATION,
            labels=[scored_label],
        ),
        Annotation(
            task_type=enums.TaskType.CLASSIFICATION,
            labels=[scored_label],
        ),
    ]

    # valid
    Prediction(datum=datum, annotations=pds)

    # test `__post_init__`
    with pytest.raises(SchemaTypeError) as e:
        Prediction(datum="datum", annotations=pds)
    assert "`datum` should be of type" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        Prediction(
            datum=datum,
            annotations=pds[0],
        )
    assert "`annotations` should be of type" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        Prediction(
            datum=datum,
            annotations=[pds[0], pds[1], "annotation"],
        )
    assert "`annotation` should be of type" in str(e)

    with pytest.raises(ValueError) as e:
        Prediction(
            datum=datum,
            annotations=[
                Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="test", value="value", score=0.8),
                        Label(key="test", value="other", score=0.1),
                    ],
                )
            ],
        )
    assert "for label key test got scores summing to 0.9" in str(e)
