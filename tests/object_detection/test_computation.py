import numpy as np

from valor_lite.object_detection.computation import (
    calculate_ranking_boundaries,
    compute_counts,
)


def test_computation_calculate_ranking_boundaries_label_mismatch_edge_case():
    """
    In v0.37.2 and earlier 'calculate_ranking_boundaries' did not factor in label matching
    when computing IOU boundaries. This lead to issues in 'compute_counts' where TP candidates
    were eliminated by IOU masking when a FP candidate from a label mismatch performed better
    in both IOU and score.

    Note that input pairs have shape (N_rows, 7)
     0: Datum ID
     1: Groundtruth ID
     2: Prediction ID
     3: Groundtruth Label ID
     4: Prediction Label ID
     5: IOU
     6: Prediction Score
    """

    ranked_pairs = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],  # skip b/c mismatched label
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.9],  # TP for IOU threshold <= 0.1
            [
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
                0.9,
                0.8,
            ],  # TP for 0.1 < IOU threshold <= 0.9
            [0.0, 0.0, 3.0, 0.0, 0.0, 0.5, 0.1],  # this row is never reached
        ]
    )

    # ranked pairs is expected to be sorted by descending score with descending IOU as tie-breaker
    iou_boundary = calculate_ranking_boundaries(ranked_pairs)
    assert (
        iou_boundary
        == np.array(
            [
                2.0,  # ineligible rows are marked with 2.0
                0.0,  # lower IOU threshold boundary for first TP candidate
                0.1,  # lower IOU threshold boundary for second TP candidate
                2.0,  # ineligle row
            ]
        )
    ).all()


def test_computation_compute_counts_ordering_edge_case():
    """
    In v0.37.2 and earlier there was a bug where the last prediction in a bin was
    selected regardless of it having the maximum score or precison.

    The PR curve is binned over 101 fixed recall points. To test this we have to first
    ensure that at least two predictions will lie within the same bin. We can do this
    by generating a single datum with a single groundtruth and having at least 2x the
    number of predictions as there are bins. To check the edge case we then test two
    variations.

    - First prediction is the only TP
    - Second prediction is the only TP

    In both cases we need to confirm that the TP is the prediction that populates the
    resulting precision-recall curve.

    Note that input pairs have shape (N_rows, 8)
     0: Datum ID
     1: Groundtruth ID
     2: Prediction ID
     3: Groundtruth Label ID
     4: Prediction Label ID
     5: IOU
     6: Prediction Score
     7: IOU boundary
    """
    N = 202
    datum_ids = np.zeros(N)
    gt_ids = np.zeros(N)
    pd_ids = np.arange(0, N)
    gt_label_ids = np.zeros(N)
    pd_label_ids = np.zeros(N)
    ious = np.zeros(N)
    scores = np.arange(N - 1, -1, -1) / (N - 1)
    iou_boundary = np.ones(N) * 2.0

    # ==== first prediction is the TP ====
    ious[0] = 1.0
    iou_boundary[0] = 0.0

    ranked_pairs = np.hstack(
        [
            datum_ids.reshape(-1, 1),
            gt_ids.reshape(-1, 1),
            pd_ids.reshape(-1, 1),
            gt_label_ids.reshape(-1, 1),
            pd_label_ids.reshape(-1, 1),
            ious.reshape(-1, 1),
            scores.reshape(-1, 1),
            iou_boundary.reshape(-1, 1),
        ]
    ).astype(np.float64)

    pr_curve = np.zeros((1, 1, 101, 2))  # updated by reference
    _ = compute_counts(
        ranked_pairs=ranked_pairs,
        iou_thresholds=np.array([0.5]),
        score_thresholds=np.array([0.5]),
        number_of_groundtruths_per_label=np.array([N]),
        number_of_labels=1,
        running_counts=np.zeros((1, 1, 2), dtype=np.uint64),
        pr_curve=pr_curve,
    )

    # test that pr curve contains highest precision and score per recall bin
    assert pr_curve.shape == (1, 1, 101, 2)
    assert pr_curve[0, 0, :, 0].tolist() == [1.0] + [0.0] * (
        100
    )  # precision computed from first row
    assert pr_curve[0, 0, :, 0].tolist() == [float(scores[0])] + [0.0] * (
        100
    )  # first score

    # ==== second prediction is the TP ====
    ious[1] = 1.0
    iou_boundary[1] = 0.0

    ranked_pairs = np.hstack(
        [
            datum_ids.reshape(-1, 1),
            gt_ids.reshape(-1, 1),
            pd_ids.reshape(-1, 1),
            gt_label_ids.reshape(-1, 1),
            pd_label_ids.reshape(-1, 1),
            ious.reshape(-1, 1),
            scores.reshape(-1, 1),
            iou_boundary.reshape(-1, 1),
        ]
    ).astype(np.float64)

    pr_curve = np.zeros((1, 1, 101, 2))  # updated by reference
    _ = compute_counts(
        ranked_pairs=ranked_pairs,
        iou_thresholds=np.array([0.5]),
        score_thresholds=np.array([0.5]),
        number_of_groundtruths_per_label=np.array([N]),
        number_of_labels=1,
        running_counts=np.zeros((1, 1, 2), dtype=np.uint64),
        pr_curve=pr_curve,
    )

    # test that pr curve contains highest precision and score per recall bin
    assert pr_curve.shape == (1, 1, 101, 2)
    assert pr_curve[0, 0, :, 0].tolist() == [1.0] + [0.0] * (
        100
    )  # precision computed from second row
    assert pr_curve[0, 0, :, 0].tolist() == [float(scores[0])] + [0.0] * (
        100
    )  # first score even though its not a TP
