import copy

import pytest

from valor import Annotation, Datum, GroundTruth, Label, Prediction, enums
from valor.schemas import Score


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
    assert "TaskType" in str(e)
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
    with pytest.raises(ValueError) as e:
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
    assert "TaskType" in str(e)
    with pytest.raises(TypeError) as e:
        Annotation(task_type=enums.TaskType.CLASSIFICATION, labels=l1)  # type: ignore
    assert "List[Label]" in str(e)
    with pytest.raises(TypeError) as e:
        Annotation(
            task_type=enums.TaskType.CLASSIFICATION, labels=[l1, l2, "label"]  # type: ignore
        )
    assert "Label" in str(e)


def test_prediction_annotation():
    l1 = Label(key="test", value="value")
    l2 = Label(key="test", value="other")
    l3 = Label(key="other", value="value")

    s1 = copy.deepcopy(l1)
    s1.score = Score(0.5)
    s2 = copy.deepcopy(l2)
    s2.score = Score(0.5)
    s3 = copy.deepcopy(l3)
    s3.score = Score(1.0)

    # valid
    Annotation(task_type=enums.TaskType.CLASSIFICATION, labels=[s1, s2, s3])

    # test `__post_init__`
    with pytest.raises(ValueError) as e:
        Annotation(task_type="something", labels=[s1, s2, s3])  # type: ignore
    assert "TaskType" in str(e)
    with pytest.raises(TypeError) as e:
        Annotation(task_type=enums.TaskType.CLASSIFICATION, labels=s1)  # type: ignore
    assert "List[Label]" in str(e)
    with pytest.raises(TypeError) as e:
        Annotation(
            task_type=enums.TaskType.CLASSIFICATION, labels=[s1, s2, "label"]  # type: ignore
        )
    assert "Label" in str(e)


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
    assert "Datum" in str(e)
    with pytest.raises(TypeError) as e:
        GroundTruth(
            datum=datum,
            annotations=gts[0],  # type: ignore
        )
    assert "List[Annotation]" in str(e)
    with pytest.raises(TypeError) as e:
        GroundTruth(
            datum=datum,
            annotations=[gts[0], gts[1], "annotation"],  # type: ignore
        )
    assert "Annotation" in str(e)

    # test equalities
    with pytest.raises(TypeError):
        _ = (
            GroundTruth(
                datum=datum,
                annotations=gts,
            )
            == 1
        )

    assert GroundTruth(datum=datum, annotations=gts,) == GroundTruth(
        datum=datum,
        annotations=gts,
    )


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

    string = str(Prediction(datum=datum, annotations=pds))
    assert (
        string
        == "{'datum': {'uid': 'somefile', 'metadata': {}}, 'annotations': [{'task_type': 'classification', 'labels': [{'key': 'test', 'value': 'value', 'score': 1.0}], 'metadata': {}, 'bounding_box': None, 'polygon': None, 'raster': None, 'embedding': None}, {'task_type': 'classification', 'labels': [{'key': 'test', 'value': 'value', 'score': 1.0}], 'metadata': {}, 'bounding_box': None, 'polygon': None, 'raster': None, 'embedding': None}]}"
    )
    assert "dataset_name" not in string

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        Prediction.create(datum="datum", annotations=pds)  # type: ignore
    assert "Datum" in str(e)
    with pytest.raises(TypeError) as e:
        Prediction.create(
            datum=datum,
            annotations=pds[0],  # type: ignore
        )
    assert "List[Annotation]" in str(e)

    with pytest.raises(TypeError) as e:
        Prediction.create(
            datum=datum,
            annotations=[pds[0], pds[1], "annotation"],  # type: ignore
        )
    assert "Annotation" in str(e)

    with pytest.raises(ValueError) as e:
        Prediction.create(
            datum=datum,
            annotations=[
                Annotation.create(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[
                        Label.create(key="test", value="value", score=0.8),
                        Label.create(key="test", value="other", score=0.1),
                    ],
                )
            ],
        )
    assert "for label key test got scores summing to 0.9" in str(e)

    # test equalities
    with pytest.raises(TypeError):
        _ = Prediction.create(datum=datum, annotations=pds) == 1

    assert Prediction.create(
        datum=datum, annotations=pds
    ) == Prediction.create(datum=datum, annotations=pds)
