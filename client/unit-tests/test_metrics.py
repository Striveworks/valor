import numpy as np


def get_tps_fps_thresholds(y_true, y_score):
    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    y_score = y_score[sorted_indices]

    distinct_value_indices = np.where(np.diff(y_score, append=0))[0]
    # threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    threshold_idxs = distinct_value_indices

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = threshold_idxs - tps + 1

    return tps, fps, y_score[threshold_idxs]


def combine_tps_fps_thresholds(
    tps1, fps1, thresholds1, tps2, fps2, thresholds2
):

    i, j, k = 0, 0, 0
    ret_length = len(thresholds1) + len(thresholds2)
    tps = np.zeros(ret_length)
    fps = np.zeros(ret_length)
    thresholds = np.zeros(ret_length)

    curr1, curr2 = 0, 0

    while k < ret_length:

        t1 = thresholds1[i] if i < len(thresholds1) else -1
        t2 = thresholds2[j] if j < len(thresholds2) else -1
        if t1 > t2:
            curr1 = tps1[i]
            thresholds[k] = t1
            if i < len(thresholds1) - 1:
                i += 1

        elif t1 == t2:
            thresholds[k] = t1
            curr1, curr2 = tps1[i], tps2[j]
            i += 1
            j += 1
        else:
            curr2 = tps2[j]
            thresholds[k] = t2
            j += 1

        tps[k] = curr1 + curr2
        k += 1

    return tps, fps, thresholds


def test_get_tps_fps_thresholds():
    y_true = np.array([1, 0, 1, 1, 0])
    y_score = np.array([0.8, 0.9, 0.7, 0.5, 0.6])

    tps, fps, thresholds = get_tps_fps_thresholds(y_true, y_score)

    np.testing.assert_equal(thresholds, np.array([0.9, 0.8, 0.7, 0.6, 0.5]))
    np.testing.assert_equal(tps, np.array([0, 1, 2, 2, 3]))
    np.testing.assert_equal(fps, np.array([1, 1, 1, 2, 2]))


def test_combine_tps_fps_thresholds():
    y_true1 = np.array([1, 1])
    y_score1 = np.array([0.8, 0.5])

    y_true2 = np.array([0, 0, 1])
    y_score2 = np.array([0.9, 0.6, 0.7])

    tps1, fps1, thresholds1 = get_tps_fps_thresholds(y_true1, y_score1)
    tps2, fps2, thresholds2 = get_tps_fps_thresholds(y_true2, y_score2)

    tps, fps, thresholds = combine_tps_fps_thresholds(
        tps1, fps1, thresholds1, tps2, fps2, thresholds2
    )

    np.testing.assert_equal(thresholds, np.array([0.9, 0.8, 0.7, 0.6, 0.5]))
    np.testing.assert_equal(tps, np.array([0, 1, 2, 2, 3]))
    # np.testing.assert_equal(fps, np.array([1, 1, 1, 2, 2]))
