import copy

import pytest

from valor import Annotation, Datum, GroundTruth, Label, Prediction
from valor.schemas import Float, Polygon


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
        bounding_box=bbox,
        labels=labels,
    )
    Annotation(
        polygon=polygon,
        labels=labels,
    )
    Annotation(raster=raster, labels=labels)
    Annotation(
        raster=raster,
        labels=labels,
    )
    Annotation(
        labels=labels,
        bounding_box=bbox,
        polygon=polygon,
        raster=raster,
    )
    Annotation(labels=labels)
    Annotation(labels=labels, metadata={})
    Annotation(
        labels=labels,
        metadata=metadata,
    )
    Annotation(
        labels=labels,
        polygon=bbox,  # bbox is a constrained polygon so this is valid usage
    )

    # test `__post_init__`
    with pytest.raises(TypeError):
        Annotation(
            labels=labels,
            bounding_box=Polygon([[(0, 0), (1, 0), (1, 1), (0, 0)]]),  # type: ignore
        )
    with pytest.raises(TypeError):
        Annotation(
            labels=labels,
            raster=bbox,
        )
    with pytest.raises(TypeError):
        Annotation(
            labels=labels,
            metadata=[1234],  # type: ignore
        )
    with pytest.raises(TypeError):
        Annotation(
            labels=labels,
            metadata={1: 1},  # type: ignore
        )
    with pytest.raises(ValueError):
        Annotation(
            labels=labels,
            metadata={"test": None},  # type: ignore
        )


def test_groundtruth_annotation():
    l1 = Label(key="test", value="value")
    l2 = Label(key="test", value="other")
    l3 = Label(key="other", value="value")

    # valid
    Annotation(
        labels=[l1, l2, l3],
    )

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        Annotation(labels=l1)  # type: ignore
    assert "List[Label]" in str(e)
    with pytest.raises(TypeError):
        Annotation(labels=[l1, l2, "label"])  # type: ignore


def test_prediction_annotation():
    l1 = Label(key="test", value="value")
    l2 = Label(key="test", value="other")
    l3 = Label(key="other", value="value")

    s1 = copy.deepcopy(l1)
    s1.score = Float.nullable(0.5)
    s2 = copy.deepcopy(l2)
    s2.score = Float.nullable(0.5)
    s3 = copy.deepcopy(l3)
    s3.score = Float.nullable(1.0)

    # valid
    Annotation(labels=[s1, s2, s3])

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        Annotation(labels=s1)  # type: ignore
    assert "List[Label]" in str(e)
    with pytest.raises(TypeError):
        Annotation(labels=[s1, s2, "label"])  # type: ignore


def test_groundtruth():
    label = Label(key="test", value="value")
    datum = Datum(uid="somefile")
    gts = [
        Annotation(labels=[label]),
        Annotation(labels=[label]),
    ]

    # valid
    GroundTruth(
        datum=datum,
        annotations=gts,
    )

    # test `__post_init__`
    with pytest.raises(TypeError):
        GroundTruth(
            datum="datum",  # type: ignore
            annotations=gts,
        )
    with pytest.raises(TypeError) as e:
        GroundTruth(
            datum=datum,
            annotations=gts[0],  # type: ignore
        )
    assert "List[Annotation]" in str(e)
    with pytest.raises(TypeError):
        GroundTruth(
            datum=datum,
            annotations=[gts[0], gts[1], "annotation"],  # type: ignore
        )

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
            labels=[scored_label],
        ),
        Annotation(
            labels=[scored_label],
        ),
    ]

    pred = Prediction(datum=datum, annotations=pds)
    string = str(pred)
    assert (
        string
        == "{'datum': {'uid': 'somefile', 'metadata': {}}, 'annotations': [{'metadata': {}, 'labels': [{'key': 'test', 'value': 'value', 'score': 1.0}], 'bounding_box': None, 'polygon': None, 'raster': None, 'embedding': None, 'is_instance_segmentation': None}, {'metadata': {}, 'labels': [{'key': 'test', 'value': 'value', 'score': 1.0}], 'bounding_box': None, 'polygon': None, 'raster': None, 'embedding': None, 'is_instance_segmentation': None}]}"
    )
    assert "dataset_name" not in string

    # test `__post_init__`
    with pytest.raises(TypeError):
        Prediction(datum="datum", annotations=pds)  # type: ignore
    with pytest.raises(TypeError) as e:
        Prediction(
            datum=datum,
            annotations=pds[0],  # type: ignore
        )
    assert "List[Annotation]" in str(e)

    with pytest.raises(TypeError):
        Prediction(
            datum=datum,
            annotations=[pds[0], pds[1], "annotation"],  # type: ignore
        )

    # test equalities
    with pytest.raises(TypeError):
        _ = Prediction(datum=datum, annotations=pds) == 1

    assert Prediction(datum=datum, annotations=pds) == Prediction(
        datum=datum, annotations=pds
    )
