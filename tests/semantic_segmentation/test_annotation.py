import numpy as np
import pytest

from valor_lite.semantic_segmentation import Loader, Segmentation


def test_segmentation():
    gt = np.array([[0, 1], [1, 0]], dtype=np.int64)
    pd = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    segmentation = Segmentation("image", gt, pd, ["sky", "road"])
    assert segmentation.shape == (2, 2)
    assert segmentation.size == 4
    assert segmentation.groundtruths.dtype == np.uint16
    assert segmentation.predictions.dtype == np.uint16
    np.testing.assert_array_equal(segmentation.groundtruths, gt)
    np.testing.assert_array_equal(segmentation.predictions, pd)
    assert gt.dtype == np.int64
    assert pd.dtype == np.uint8


@pytest.mark.parametrize("side", ["groundtruths", "predictions"])
@pytest.mark.parametrize(
    "invalid",
    [
        [[0]],
        np.array([[True]]),
        np.array([[0.0]]),
        np.array([[0j]]),
        np.array([[0]], dtype=object),
        np.array([0]),
        np.zeros((1, 1, 1), dtype=np.uint16),
        np.zeros((0, 1), dtype=np.uint16),
        np.array([[-1]]),
        np.array([[2]]),
        np.array([[65536]], dtype=np.uint64),
        np.array([[2**64 - 1]], dtype=np.uint64),
    ],
)
def test_invalid_maps(side, invalid):
    kwargs = {
        "groundtruths": np.zeros((1, 1), dtype=np.uint16),
        "predictions": np.zeros((1, 1), dtype=np.uint16),
    }
    kwargs[side] = invalid
    with pytest.raises(ValueError):
        Segmentation(
            "image",
            groundtruths=kwargs["groundtruths"],
            predictions=kwargs["predictions"],
            labels=["sky"],
        )


def test_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        Segmentation(
            "image",
            np.zeros((1, 2), dtype=int),
            np.zeros((2, 1), dtype=int),
            ["sky"],
        )


@pytest.mark.parametrize(
    "labels",
    [["sky", "sky"], [0], "sky", None, [str(i) for i in range(65536)]],
)
def test_invalid_labels(labels):
    with pytest.raises(ValueError):
        Segmentation(
            "image",
            np.zeros((1, 1), dtype=int),
            np.zeros((1, 1), dtype=int),
            labels,
        )


def test_highest_label_id():
    array = np.array([[0, 65535]], dtype=np.uint64)
    segmentation = Segmentation(
        "image", array, array, [str(i) for i in range(1, 65536)]
    )
    np.testing.assert_array_equal(segmentation.groundtruths, array)
    assert segmentation.groundtruths.dtype == np.uint16


@pytest.mark.parametrize(
    "side", ["groundtruth_metadata", "prediction_metadata"]
)
@pytest.mark.parametrize(
    "metadata", [{-1: {}}, {2: {}}, {"0": {}}, {True: {}}, {0: "invalid"}, []]
)
def test_invalid_class_metadata(side, metadata):
    with pytest.raises(ValueError, match=side):
        Segmentation(
            "image",
            np.zeros((1, 1), dtype=int),
            np.zeros((1, 1), dtype=int),
            ["sky"],
            **{side: metadata}
        )


def test_read_only_noncontiguous_arrays_are_not_modified():
    array = np.arange(12, dtype=np.uint16).reshape(3, 4) % 2
    original = array.copy()
    array.flags.writeable = False
    gt = array[:, ::2]
    pd = array[:, 1::2]
    segmentation = Segmentation("image", gt, pd, ["sky", "road"])
    loader = Loader.in_memory()
    loader.add_data([segmentation])
    evaluator = loader.finalize()
    np.testing.assert_array_equal(array, original)
    assert evaluator.info.number_of_pixels == 6
    assert evaluator._compute_confusion_matrix_intermediate()[0, 1] == 6


def test_empty_labels_require_all_background():
    background = np.zeros((2, 2), dtype=np.uint16)
    segmentation = Segmentation("image", background, background, [])
    assert segmentation.labels == []
    with pytest.raises(ValueError, match="len\\(labels\\)"):
        Segmentation("image", background, np.ones((2, 2), dtype=np.uint16), [])
