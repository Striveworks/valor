from dataclasses import asdict

import pytest

from velour.data_types import (
    BoundingBox,
    BoundingPolygon,
    GroundTruthDetection,
    GroundTruthInstanceSegmentation,
    GroundTruthSemanticSegmentation,
    Label,
    PredictedDetection,
    PredictedImageClassification,
    ScoredLabel,
    _GroundTruthSegmentation,
)


def test_ground_truth_segmentation():
    """Test that a _GroundTruthSegmentation object can't be instantiated"""
    with pytest.raises(TypeError) as exc_info:
        _GroundTruthSegmentation(
            shape=None, labels=None, image=None, _is_instance=None
        )

    assert "Cannot instantiate abstract class" in str(exc_info)


def test_ground_truth_instance_segmentation():
    """Test that _is_instance can't be passed to the constructor and
    that when not passing, _is_instance gets set to True
    """
    with pytest.raises(TypeError) as exc_info:
        GroundTruthInstanceSegmentation(
            shape=None, labels=None, image=None, _is_instance=True
        )

    assert "got an unexpected keyword argument" in str(exc_info)

    seg = GroundTruthInstanceSegmentation(shape=None, labels=None, image=None)
    assert seg._is_instance


def test_ground_truth_semantic_segmentation():
    """Test that _is_instance can't be passed to the constructor and
    that when not passing, _is_instance gets set to False
    """
    with pytest.raises(TypeError) as exc_info:
        GroundTruthSemanticSegmentation(
            shape=None, labels=None, image=None, _is_instance=True
        )

    assert "got an unexpected keyword argument" in str(exc_info)

    seg = GroundTruthSemanticSegmentation(shape=None, labels=None, image=None)
    assert not seg._is_instance


def test_bounding_box():
    with pytest.raises(ValueError) as exc_info:
        BoundingBox(xmin=0, ymin=10, xmax=100, ymax=5)
    assert "Cannot have ymin > ymax" in str(exc_info)

    with pytest.raises(ValueError) as exc_info:
        BoundingBox(xmin=10, ymin=10, xmax=0, ymax=50)
    assert "Cannot have xmin > xmax" in str(exc_info)

    kwargs = dict(xmin=10, ymin=10, xmax=40, ymax=50)
    bbox = BoundingBox(**kwargs)
    assert asdict(bbox) == kwargs


def test_ground_truth_detection():
    with pytest.raises(ValueError) as exc_info:
        GroundTruthDetection(image=None, labels=[])
    assert "Must pass exactly one of `boundary` or `bbox`" in str(exc_info)

    with pytest.raises(ValueError) as exc_info:
        GroundTruthDetection(
            image=None,
            labels=[],
            boundary=BoundingPolygon([]),
            bbox=BoundingBox(1, 2, 3, 4),
        )
    assert "Must pass exactly one of `boundary` or `bbox`" in str(exc_info)

    assert GroundTruthDetection(
        image=None, labels=[], bbox=BoundingBox(1, 2, 3, 4)
    ).is_bbox

    assert not GroundTruthDetection(
        image=None,
        labels=[],
        boundary=BoundingPolygon([]),
    ).is_bbox


def test_predicted_detection():
    with pytest.raises(ValueError) as exc_info:
        PredictedDetection(image=None, scored_labels=[])
    assert "Must pass exactly one of `boundary` or `bbox`" in str(exc_info)

    with pytest.raises(ValueError) as exc_info:
        PredictedDetection(
            image=None,
            scored_labels=[],
            boundary=BoundingPolygon([]),
            bbox=BoundingBox(1, 2, 3, 4),
        )
    assert "Must pass exactly one of `boundary` or `bbox`" in str(exc_info)

    assert PredictedDetection(
        image=None, scored_labels=[], bbox=BoundingBox(1, 2, 3, 4)
    )

    assert PredictedDetection(
        image=None,
        scored_labels=[],
        boundary=BoundingPolygon([]),
    )


def gest_predicted_classification():
    with pytest.raises(ValueError) as exc_info:
        PredictedImageClassification(
            image=None,
            scored_labels=[
                ScoredLabel(label=Label("k", "v1"), score=0.2),
                ScoredLabel(label=Label("k", "v2"), score=0.7),
            ],
        )
    assert "must sum to 1" in str(exc_info)

    assert PredictedImageClassification(
        image=None,
        scored_labels=[
            ScoredLabel(label=Label("k", "v1"), score=0.2),
            ScoredLabel(label=Label("k", "v2"), score=0.8),
        ],
    )
