import pytest
from PIL import Image

from velour.data_types import (
    GroundTruthInstanceSegmentation,
    GroundTruthSegmentation,
    GroundTruthSemanticSegmentation,
    coco_rle_to_mask,
)


def test_coco_rle_to_mask():
    h, w = 4, 6
    coco_rle_seg_dict = {"counts": [10, 4, 3, 7], "size": (h, w)}

    mask = coco_rle_to_mask(coco_rle_seg_dict=coco_rle_seg_dict)
    expected_mask = [
        [False, False, False, True, False, True],
        [False, False, False, True, True, True],
        [False, False, True, False, True, True],
        [False, False, True, False, True, True],
    ]

    assert mask.sum() == 4 + 7
    assert mask.tolist() == expected_mask

    img = Image.fromarray(mask)

    assert img.width == w
    assert img.height == h


def test_ground_truth_segmentation():
    """Test that a GroundTruthSegmentation object can't be instantiated"""
    with pytest.raises(TypeError) as exc_info:
        GroundTruthSegmentation(
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
