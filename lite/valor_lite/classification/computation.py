import numpy as np
from numpy.typing import NDArray

# 0 datum index
# 1 gt label index
# 2 pd label index
# 3 score


def compute_metrics(
    data: NDArray[np.floating],
    label_metadata: NDArray[np.int32],
    score_thresholds: NDArray[np.floating],
) -> tuple:

    n_labels = label_metadata.shape[0]
    n_scores = score_thresholds.shape[0]

    pd_labels = data[:, 2].astype(int)

    mask_gt_exists = data[:, 1] >= 0.0
    mask_pd_exists = data[:, 2] >= 0.0
    mask_matching_labels = np.isclose(data[:, 1], data[:, 2])
    mask_score_nonzero = ~np.isclose(data[:, 3], 0.0)

    for score_idx in range(n_scores):
        mask_score_threshold = data[:, 3] >= score_thresholds[score_idx]
        mask_score = mask_score_nonzero & mask_score_threshold

        mask_tp = (
            mask_gt_exists & mask_pd_exists & mask_matching_labels & mask_score
        )
        mask_fp = (
            mask_pd_exists
            & (~mask_gt_exists | ~mask_matching_labels)
            & mask_score
        )
        mask_fn = mask_gt_exists & (
            (mask_pd_exists & mask_matching_labels & ~mask_score)
            | ~mask_pd_exists
        )
        mask_tn = (
            mask_pd_exists
            & mask_gt_exists
            & ~mask_matching_labels
            & ~mask_score
        )

        tp_count = np.bincount(pd_labels[mask_tp], minlength=n_labels)
        fp_count = np.bincount(pd_labels[mask_fp], minlength=n_labels)
        fn_count = np.bincount(pd_labels[mask_fn], minlength=n_labels)
        tn_count = np.bincount(pd_labels[mask_tn], minlength=n_labels)

    return (
        tp_count,
        tn_count,
        fp_count,
        fn_count,
    )


def compute_detailed_pr_curve() -> tuple:
    return tuple()
