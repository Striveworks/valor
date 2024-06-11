from typing import Tuple

import numpy as np


def get_tps_fps_thresholds(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(y_true) != len(y_score):
        raise ValueError(
            "y_true and y_score must have the same length, but got "
            f"{len(y_true)} and {len(y_score)}"
        )
    if y_true.dtype != bool:
        raise ValueError("y_true must be a boolean array")
    if y_score.dtype != float or y_score.min() < 0 or y_score.max() > 1:
        raise ValueError("y_score must be a float array in the range [0, 1]")

    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    y_score = y_score[sorted_indices]

    distinct_value_indices = np.where(np.diff(y_score, append=0))[0]

    tps = np.cumsum(y_true)[distinct_value_indices]
    fps = distinct_value_indices - tps + 1

    return tps, fps, y_score[distinct_value_indices]


def combine_tps_fps_thresholds(
    tps1: np.ndarray,
    fps1: np.ndarray,
    thresholds1: np.ndarray,
    tps2: np.ndarray,
    fps2: np.ndarray,
    thresholds2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    i, j, k = 0, 0, 0
    ret_length = len(thresholds1) + len(thresholds2)
    tps = np.zeros(ret_length)
    fps = np.zeros(ret_length)
    thresholds = np.zeros(ret_length)

    curr1, curr2 = 0, 0
    curr_fp1, curr_fp2 = 0, 0
    dups = 0

    while k < ret_length - dups:
        t1 = thresholds1[i] if i < len(thresholds1) else -1
        t2 = thresholds2[j] if j < len(thresholds2) else -1
        if t1 > t2:
            curr1 = tps1[i]
            curr_fp1 = fps1[i]
            thresholds[k] = t1
            if i < len(thresholds1) - 1:
                i += 1
        elif t1 == t2:
            thresholds[k] = t1
            curr1, curr2 = tps1[i], tps2[j]
            curr_fp1, curr_fp2 = fps1[i], fps2[j]
            dups += 1
            i += 1
            j += 1
        else:
            curr2 = tps2[j]
            curr_fp2 = fps2[j]
            thresholds[k] = t2
            j += 1

        tps[k] = curr1 + curr2
        fps[k] = curr_fp1 + curr_fp2
        k += 1

    return (
        tps[: ret_length - dups],
        fps[: ret_length - dups],
        thresholds[: ret_length - dups],
    )
