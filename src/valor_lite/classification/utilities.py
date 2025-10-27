from collections import defaultdict

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from valor_lite.classification.metric import Metric, MetricType


def unpack_precision_recall_rocauc_into_metric_lists(
    counts: NDArray[np.uint64],
    precision: NDArray[np.float64],
    recall: NDArray[np.float64],
    accuracy: NDArray[np.float64],
    f1_score: NDArray[np.float64],
    rocauc: NDArray[np.float64],
    mean_rocauc: float,
    score_thresholds: list[float],
    hardmax: bool,
    index_to_label: dict[int, str],
) -> dict[MetricType, list[Metric]]:

    metrics = defaultdict(list)

    metrics[MetricType.ROCAUC] = [
        Metric.roc_auc(
            value=float(rocauc[label_idx]),
            label=label,
        )
        for label_idx, label in index_to_label.items()
    ]

    metrics[MetricType.mROCAUC] = [
        Metric.mean_roc_auc(
            value=float(mean_rocauc),
        )
    ]

    metrics[MetricType.Accuracy] = [
        Metric.accuracy(
            value=float(accuracy[score_idx]),
            score_threshold=score_threshold,
            hardmax=hardmax,
        )
        for score_idx, score_threshold in enumerate(score_thresholds)
    ]

    for label_idx, label in index_to_label.items():
        for score_idx, score_threshold in enumerate(score_thresholds):

            kwargs = {
                "label": label,
                "hardmax": hardmax,
                "score_threshold": score_threshold,
            }
            row = counts[:, label_idx]
            metrics[MetricType.Counts].append(
                Metric.counts(
                    tp=int(row[score_idx, 0]),
                    fp=int(row[score_idx, 1]),
                    fn=int(row[score_idx, 2]),
                    tn=int(row[score_idx, 3]),
                    **kwargs,
                )
            )

            metrics[MetricType.Precision].append(
                Metric.precision(
                    value=float(precision[score_idx, label_idx]),
                    **kwargs,
                )
            )
            metrics[MetricType.Recall].append(
                Metric.recall(
                    value=float(recall[score_idx, label_idx]),
                    **kwargs,
                )
            )
            metrics[MetricType.F1].append(
                Metric.f1_score(
                    value=float(f1_score[score_idx, label_idx]),
                    **kwargs,
                )
            )
    return metrics


def unpack_confusion_matrix(
    confusion_matrices: NDArray[np.uint64],
    unmatched_groundtruths: NDArray[np.uint64],
    index_to_label: dict[int, str],
    score_thresholds: list[float],
    hardmax: bool,
) -> list[Metric]:
    metrics = []
    for score_idx, score_thresh in enumerate(score_thresholds):
        cm_dict = {}
        ugt_dict = {}
        for idx, label in index_to_label.items():
            ugt_dict[label] = int(unmatched_groundtruths[score_idx, idx])
            for pidx, plabel in index_to_label.items():
                if label not in cm_dict:
                    cm_dict[label] = {}
                cm_dict[label][plabel] = int(
                    confusion_matrices[score_idx, idx, pidx]
                )
        metrics.append(
            Metric.confusion_matrix(
                confusion_matrix=cm_dict,
                unmatched_ground_truths=ugt_dict,
                score_threshold=score_thresh,
                hardmax=hardmax,
            )
        )
    return metrics


def create_mapping(
    tbl: pa.Table,
    pairs: NDArray[np.float64],
    index: int,
    id_col: str,
    uid_col: str,
) -> dict[int, str]:
    col = pairs[:, index].astype(np.int64)
    values, indices = np.unique(col, return_index=True)
    indices = indices[values >= 0]
    return {
        tbl[id_col][idx].as_py(): tbl[uid_col][idx].as_py() for idx in indices
    }


def unpack_examples(
    ids: NDArray[np.int64],
    mask_tp: NDArray[np.bool_],
    mask_fn: NDArray[np.bool_],
    mask_fp: NDArray[np.bool_],
    score_thresholds: list[float],
    hardmax: bool,
    index_to_datum_id: dict[int, str],
    index_to_label: dict[int, str],
) -> list[Metric]:
    metrics = []
    unique_datums = np.unique(ids[:, 0])
    for datum_index in unique_datums:
        mask_datum = ids[:, 0] == datum_index
        mask_datum_tp = mask_tp & mask_datum
        mask_datum_fp = mask_fp & mask_datum
        mask_datum_fn = mask_fn & mask_datum

        datum_id = index_to_datum_id[datum_index]
        for score_idx, score_thresh in enumerate(score_thresholds):

            unique_tp = np.unique(
                ids[np.ix_(mask_datum_tp[score_idx], (0, 1, 2))], axis=0  # type: ignore - numpy ix_ typing
            )
            unique_fp = np.unique(
                ids[np.ix_(mask_datum_fp[score_idx], (0, 2))], axis=0  # type: ignore - numpy ix_ typing
            )
            unique_fn = np.unique(
                ids[np.ix_(mask_datum_fn[score_idx], (0, 1))], axis=0  # type: ignore - numpy ix_ typing
            )

            tp = [index_to_label[row[1]] for row in unique_tp]
            fp = [
                index_to_label[row[1]]
                for row in unique_fp
                if index_to_label[row[1]] not in tp
            ]
            fn = [
                index_to_label[row[1]]
                for row in unique_fn
                if index_to_label[row[1]] not in tp
            ]
            metrics.append(
                Metric.examples(
                    datum_id=datum_id,
                    true_positives=tp,
                    false_negatives=fn,
                    false_positives=fp,
                    score_threshold=score_thresh,
                    hardmax=hardmax,
                )
            )
    return metrics


def create_empty_confusion_matrix_with_examples(
    score_threshold: float,
    hardmax: bool,
    index_to_label: dict[int, str],
) -> Metric:
    unmatched_groundtruths = dict()
    confusion_matrix = dict()
    for label in index_to_label.values():
        unmatched_groundtruths[label] = {"count": 0, "examples": []}
        confusion_matrix[label] = {}
        for plabel in index_to_label.values():
            confusion_matrix[label][plabel] = {"count": 0, "examples": []}

    return Metric.confusion_matrix_with_examples(
        confusion_matrix=confusion_matrix,
        unmatched_ground_truths=unmatched_groundtruths,
        score_threshold=score_threshold,
        hardmax=hardmax,
    )


def _unpack_confusion_matrix_with_examples(
    metric: Metric,
    ids: NDArray[np.int64],
    scores: NDArray[np.float64],
    winners: NDArray[np.bool_],
    mask_matched: NDArray[np.bool_],
    mask_unmatched_fn: NDArray[np.bool_],
    index_to_datum_id: dict[int, str],
    index_to_label: dict[int, str],
):
    if not isinstance(metric.value, dict):
        raise TypeError("expected metric to contain a dictionary value")

    unique_matches, unique_match_indices = np.unique(
        ids[np.ix_(mask_matched, (0, 1, 2))],  # type: ignore - numpy ix_ typing
        axis=0,
        return_index=True,
    )
    scores = scores[mask_matched][unique_match_indices]
    unique_unmatched_groundtruths = np.unique(
        ids[np.ix_(mask_unmatched_fn, (0, 1))],  # type: ignore - numpy ix_ typing
        axis=0,
    )

    n_matched = unique_matches.shape[0]
    n_unmatched_groundtruths = unique_unmatched_groundtruths.shape[0]
    n_max = max(n_matched, n_unmatched_groundtruths)

    for idx in range(n_max):
        if idx < n_matched:
            datum_id = index_to_datum_id[unique_matches[idx, 0]]
            glabel = index_to_label[unique_matches[idx, 1]]
            plabel = index_to_label[unique_matches[idx, 2]]
            score = float(scores[idx])

            metric.value["confusion_matrix"][glabel][plabel]["count"] += 1
            metric.value["confusion_matrix"][glabel][plabel][
                "examples"
            ].append(
                {
                    "datum_id": datum_id,
                    "score": score,
                }
            )
        if idx < n_unmatched_groundtruths:
            datum_id = index_to_datum_id[unique_unmatched_groundtruths[idx, 0]]
            label = index_to_label[unique_unmatched_groundtruths[idx, 1]]

            metric.value["unmatched_ground_truths"][label]["count"] += 1
            metric.value["unmatched_ground_truths"][label]["examples"].append(
                {"datum_id": datum_id}
            )

    return metric


def unpack_confusion_matrix_with_examples(
    metrics: dict[int, Metric],
    ids: NDArray[np.int64],
    scores: NDArray[np.float64],
    winners: NDArray[np.bool_],
    mask_matched: NDArray[np.bool_],
    mask_unmatched_fn: NDArray[np.bool_],
    index_to_datum_id: dict[int, str],
    index_to_label: dict[int, str],
) -> list[Metric]:
    return [
        _unpack_confusion_matrix_with_examples(
            metric,
            ids=ids,
            scores=scores,
            winners=winners,
            mask_matched=mask_matched[score_idx, :],
            mask_unmatched_fn=mask_unmatched_fn[score_idx, :],
            index_to_datum_id=index_to_datum_id,
            index_to_label=index_to_label,
        )
        for score_idx, metric in metrics.items()
    ]
