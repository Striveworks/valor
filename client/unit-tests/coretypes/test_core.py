import copy

import pytest

from valor import Annotation, Datum, GroundTruth, Label, Prediction, enums


def test_datum():
    Datum(uid="123")
    Datum(uid="123", metadata={})
    Datum(uid="123", metadata={"name": 1})

    # test `__post_init__`
    with pytest.raises(TypeError):
        Datum(uid=123)  # type: ignore
    with pytest.raises(TypeError):
        Datum(uid="123", metadata=1)  # type: ignore
    with pytest.raises(TypeError):
        Datum(uid="123", metadata=[1])  # type: ignore


def test_annotation(bbox, polygon, raster, labels, metadata):
    # valid
    Annotation(
        task_type=enums.TaskType.OBJECT_DETECTION,
        bounding_box=bbox,
        labels=labels,
    )
    Annotation(
        task_type=enums.TaskType.OBJECT_DETECTION,
        polygon=polygon,
        labels=labels,
    )
    Annotation(
        task_type=enums.TaskType.OBJECT_DETECTION, raster=raster, labels=labels
    )
    Annotation(
        task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
        raster=raster,
        labels=labels,
    )
    Annotation(
        task_type=enums.TaskType.OBJECT_DETECTION,
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
        Annotation(task_type="something", labels=labels)  # type: ignore
    assert "is not a valid TaskType" in str(e)
    with pytest.raises(TypeError) as e:
        Annotation(
            task_type=enums.TaskType.OBJECT_DETECTION,
            labels=labels,
            bounding_box=polygon,
        )
    with pytest.raises(TypeError) as e:
        Annotation(
            task_type=enums.TaskType.OBJECT_DETECTION,
            labels=labels,
            polygon=bbox,
        )
    with pytest.raises(TypeError) as e:
        Annotation(
            task_type=enums.TaskType.OBJECT_DETECTION,
            labels=labels,
            multipolygon=bbox,
        )
    with pytest.raises(TypeError) as e:
        Annotation(
            task_type=enums.TaskType.OBJECT_DETECTION,
            labels=labels,
            raster=bbox,
        )
    with pytest.raises(TypeError) as e:
        Annotation(
            task_type=enums.TaskType.CLASSIFICATION,
            labels=labels,
            metadata=[1234],  # type: ignore
        )
    with pytest.raises(TypeError) as e:
        Annotation(
            task_type=enums.TaskType.CLASSIFICATION,
            labels=labels,
            metadata={1: 1},  # type: ignore
        )
    with pytest.raises(TypeError) as e:
        Annotation(
            task_type=enums.TaskType.CLASSIFICATION,
            labels=labels,
            metadata={"test": None},  # type: ignore
        )


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
        Annotation(task_type="soemthing", labels=[l1])  # type: ignore
    assert "is not a valid TaskType" in str(e)
    with pytest.raises(TypeError) as e:
        Annotation(task_type=enums.TaskType.CLASSIFICATION, labels=l1)  # type: ignore
    assert "List[valor.Label]" in str(e)
    with pytest.raises(TypeError) as e:
        Annotation(
            task_type=enums.TaskType.CLASSIFICATION, labels=[l1, l2, "label"]  # type: ignore
        )
    assert "valor.Label" in str(e)


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
        Annotation(task_type="something", labels=[s1, s2, s3])  # type: ignore
    assert "is not a valid TaskType" in str(e)
    with pytest.raises(TypeError) as e:
        Annotation(task_type=enums.TaskType.CLASSIFICATION, labels=s1)  # type: ignore
    assert "List[valor.Label]" in str(e)
    with pytest.raises(TypeError) as e:
        Annotation(
            task_type=enums.TaskType.CLASSIFICATION, labels=[s1, s2, "label"]  # type: ignore
        )
    assert "valor.Label" in str(e)


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
    with pytest.raises(TypeError) as e:
        GroundTruth(
            datum="datum",  # type: ignore
            annotations=gts,
        )
    assert "valor.Datum" in str(e)
    with pytest.raises(TypeError) as e:
        GroundTruth(
            datum=datum,
            annotations=gts[0],  # type: ignore
        )
    assert "List[valor.Annotation]" in str(e)
    with pytest.raises(TypeError) as e:
        GroundTruth(
            datum=datum,
            annotations=[gts[0], gts[1], "annotation"],  # type: ignore
        )
    assert "valor.Annotation" in str(e)


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
    with pytest.raises(TypeError) as e:
        Prediction(datum="datum", annotations=pds)  # type: ignore
    assert "valor.Datum" in str(e)
    with pytest.raises(TypeError) as e:
        Prediction(
            datum=datum,
            annotations=pds[0],  # type: ignore
        )
    assert "List[valor.Annotation]" in str(e)
    with pytest.raises(TypeError) as e:
        Prediction(
            datum=datum,
            annotations=[pds[0], pds[1], "annotation"],  # type: ignore
        )
    assert "valor.Annotation" in str(e)

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
