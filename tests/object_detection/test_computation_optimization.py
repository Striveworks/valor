import numpy as np
import pyarrow as pa
import pytest

from valor_lite.cache import MemoryCacheWriter
from valor_lite.object_detection.computation import (
    _encode_keys,
    compute_confusion_matrix_from_table,
    compute_counts,
)
from valor_lite.object_detection.shared import (
    extract_groundtruth_count_per_label,
)


def test_counts_score_ties_and_repeated_unsorted_thresholds():
    pairs = np.array(
        [
            [0, 0, 0, 0, 0, 0.8, 0.9, 0],
            [0, 0, 1, 0, 0, 0.7, 0.9, 0],
            [0, -1, 2, -1, 1, 0, 0.7, 2],
            [0, 1, 3, 1, 1, 0.6, 0.5, 0],
            [0, -1, 4, -1, 1, 0, 0, 2],
        ],
        dtype=np.float64,
    )
    running = np.zeros((2, 2, 2), dtype=np.uint64)
    curve = np.zeros((2, 2, 101, 2))
    counts = compute_counts(
        ranked_pairs=pairs,
        iou_thresholds=np.array([0.5, 0.75]),
        score_thresholds=np.array([0.9, 0.5, 0.9, 1.0]),
        number_of_groundtruths_per_label=np.array([2, 2], dtype=np.uint64),
        number_of_labels=2,
        running_counts=running,
        pr_curve=curve,
    )
    np.testing.assert_array_equal(
        counts[:, :, :2],
        [
            [
                [[1, 0], [1, 0]],
                [[1, 1], [1, 1]],
                [[1, 0], [1, 0]],
                [[0, 0], [0, 0]],
            ],
            [
                [[1, 0], [1, 0]],
                [[1, 0], [1, 2]],
                [[1, 0], [1, 0]],
                [[0, 0], [0, 0]],
            ],
        ],
    )
    np.testing.assert_array_equal(
        running, [[[2, 2], [3, 1]], [[2, 1], [3, 0]]]
    )


def test_counts_empty_chunk_preserves_running_state():
    running = np.ones((1, 2, 2), dtype=np.uint64)
    curve = np.ones((1, 2, 101, 2))
    counts = compute_counts(
        np.empty((0, 8)),
        np.array([0.5]),
        np.array([0.5]),
        np.array([1, 1], dtype=np.uint64),
        2,
        running,
        curve,
    )
    assert not counts.any()
    assert (running == 1).all()
    assert (curve == 1).all()


@pytest.mark.parametrize("scores", [[0.9, 0.5, np.nan], [np.nan, 0.9, 0.5]])
def test_counts_nan_score_preserves_first_score_in_recall_bin(scores):
    pairs = np.zeros((3, 8), dtype=np.float64)
    pairs[:, 1] = -1
    pairs[:, 2] = np.arange(3)
    pairs[:, 3] = -1
    pairs[:, 6] = scores
    pairs[:, 7] = 2
    curve = np.zeros((1, 1, 101, 2))
    counts = compute_counts(
        pairs,
        np.array([0.5]),
        np.array([0.5]),
        np.array([1], dtype=np.uint64),
        1,
        np.zeros((1, 1, 2), dtype=np.uint64),
        curve,
    )
    np.testing.assert_equal(curve[0, 0, 0, 1], scores[0])
    np.testing.assert_array_equal(counts[0, 0, :, 0], [0, 2, 0])


def test_confusion_greedy_matching_before_thresholds_with_large_ids():
    # A low-IoU pair wins greedy matching by score. Removing it by threshold
    # must not rematch its ground truth with the lower-score, higher-IoU pair.
    base = 2**62
    table = pa.table(
        {
            "gt_id": [base + 10, base + 11, base + 10, base + 10],
            "pd_id": [base + 20, base + 20, base + 21, base + 20],
            "gt_label_id": [0, 1, 0, 0],
            "pd_label_id": [0, 0, 1, 1],
            "iou": [0.4, 0.4, 1.0, 0.4],
            "pd_score": [0.9, 0.9, 0.8, 0.2],
        }
    )
    matrices, groundtruths, predictions = compute_confusion_matrix_from_table(
        table, 2, np.array([0.3, 0.5]), np.array([0.1, 0.5])
    )
    np.testing.assert_array_equal(
        matrices,
        [
            [[[1, 1], [0, 0]], [[1, 0], [0, 0]]],
            [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        ],
    )
    np.testing.assert_array_equal(
        groundtruths, [[[0, 1], [0, 1]], [[1, 1], [1, 1]]]
    )
    np.testing.assert_array_equal(
        predictions, [[[0, 1], [0, 1]], [[1, 2], [1, 1]]]
    )


def test_tuple_keys_do_not_collide_when_integer_packing_would_overflow():
    columns = np.array(
        [[-1, -1], [2**62, 0], [0, 2**62], [2**62, 0]], dtype=np.int64
    )
    keys = _encode_keys(columns[:, 0], columns[:, 1])
    assert len(np.unique(keys)) == 3
    assert keys[1] == keys[3]


def test_groundtruth_counts_with_large_ids_and_repeated_pairs():
    schema = pa.schema([("gt_id", pa.int64()), ("gt_label_id", pa.int64())])
    writer = MemoryCacheWriter.create(schema=schema, batch_size=10)
    writer.write_table(
        pa.table(
            {
                "gt_id": [2**62, 2**62, 2**62 + 1, -1],
                "gt_label_id": [2, 2, 3, -1],
            },
            schema=schema,
        )
    )
    counts = extract_groundtruth_count_per_label(writer.to_reader(), 4)
    np.testing.assert_array_equal(counts, [0, 0, 1, 1])
