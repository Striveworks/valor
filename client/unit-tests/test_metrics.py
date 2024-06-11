import numpy as np

from valor.metrics.classification import (
    combine_tps_fps_thresholds,
    get_tps_fps_thresholds,
)


def test_get_tps_fps_thresholds():
    y_true = np.array([True, False, True, True, False])
    y_score = np.array([0.8, 0.9, 0.7, 0.5, 0.6])

    tps, fps, thresholds = get_tps_fps_thresholds(y_true, y_score)

    np.testing.assert_equal(thresholds, np.array([0.9, 0.8, 0.7, 0.6, 0.5]))
    np.testing.assert_equal(tps, np.array([0, 1, 2, 2, 3]))
    np.testing.assert_equal(fps, np.array([1, 1, 1, 2, 2]))


def test_combine_tps_fps_thresholds():
    y_true1 = np.array([True, True])
    y_score1 = np.array([0.8, 0.5])

    y_true2 = np.array([False, False, True])
    y_score2 = np.array([0.9, 0.6, 0.7])

    tps1, fps1, thresholds1 = get_tps_fps_thresholds(y_true1, y_score1)
    tps2, fps2, thresholds2 = get_tps_fps_thresholds(y_true2, y_score2)

    tps, fps, thresholds = combine_tps_fps_thresholds(
        tps1, fps1, thresholds1, tps2, fps2, thresholds2
    )

    np.testing.assert_equal(thresholds, np.array([0.9, 0.8, 0.7, 0.6, 0.5]))
    np.testing.assert_equal(tps, np.array([0, 1, 2, 2, 3]))
    np.testing.assert_equal(fps, np.array([1, 1, 1, 2, 2]))


def test_combine_tps_fps_thresholds_dup_threshold():
    y_true1 = np.array([True, True])
    y_score1 = np.array([0.8, 0.5])

    y_true2 = np.array([False, False, True])
    y_score2 = np.array([0.9, 0.5, 0.7])

    tps1, fps1, thresholds1 = get_tps_fps_thresholds(y_true1, y_score1)
    tps2, fps2, thresholds2 = get_tps_fps_thresholds(y_true2, y_score2)

    tps, fps, thresholds = combine_tps_fps_thresholds(
        tps1, fps1, thresholds1, tps2, fps2, thresholds2
    )

    np.testing.assert_equal(thresholds, np.array([0.9, 0.8, 0.7, 0.5]))
    np.testing.assert_equal(tps, np.array([0, 1, 2, 3]))
    np.testing.assert_equal(fps, np.array([1, 1, 1, 2]))
