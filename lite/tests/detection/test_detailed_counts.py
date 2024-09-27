import numpy as np
from valor_lite.detection import DataLoader, Detection, Evaluator, MetricType
from valor_lite.detection.computation import compute_detailed_metrics


def test_counts_with_examples_no_data():
    evaluator = Evaluator()
    curves = evaluator._compute_detailed_metrics()
    assert isinstance(curves, list)
    assert len(curves) == 0


def test_compute_counts_with_examples():
    sorted_pairs = np.array(
        [
            # dt,  gt,  pd,  iou,  gl,  pl, score,
            [0.0, 0.0, 1.0, 0.98, 0.0, 0.0, 0.9],
            [1.0, 1.0, 2.0, 0.55, 1.0, 0.0, 0.9],
            [2.0, -1.0, 4.0, 0.0, -1.0, 0.0, 0.65],
            [3.0, 4.0, 5.0, 1.0, 0.0, 0.0, 0.1],
            [1.0, 1.0, 3.0, 0.55, 0.0, 0.0, 0.1],
            [4.0, 5.0, -1.0, 0.0, 0.0, -1.0, -1.0],
        ]
    )
    label_metadata = np.array([[3, 4], [1, 0]])
    iou_thresholds = np.array([0.5])
    score_thresholds = np.array([score / 100.0 for score in range(1, 101)])

    results = compute_detailed_metrics(
        data=sorted_pairs,
        label_metadata=label_metadata,
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
        n_examples=0,
    )

    assert len(results) == 1
    assert results.shape == (1, 100, 2, 5)  # iou, score, label, metrics

    """
    @ iou=0.5, score<0.1
    3x tp
    1x fp misclassification
    1x fp hallucination
    0x fn misclassification
    1x fn missing prediction
    """
    assert np.isclose(results[0, :10, 0, :], np.array([3, 1, 1, 0, 1])).all()

    """
    @ iou=0.5, 0.1 <= score < 0.65
    1x tp
    1x fp misclassification
    1x fp hallucination
    1x fn misclassification
    2x fn missing prediction
    """
    assert np.isclose(results[0, 10:65, 0, :], np.array([1, 1, 1, 1, 2])).all()

    """
    @ iou=0.5, 0.65 <= score < 0.9
    1x tp
    1x fp misclassification
    0x fp hallucination
    1x fn misclassification
    2x fn missing prediction
    """
    assert np.isclose(results[0, 65:90, 0, :], np.array([1, 1, 0, 1, 2])).all()

    """
    @ iou=0.5, score>=0.9
    0x tp
    0x fp misclassification
    0x fp hallucination
    0x fn misclassification
    4x fn missing prediction
    """
    assert np.isclose(results[0, 90:, 0, :], np.array([0, 0, 0, 0, 4])).all()

    # compute with examples

    """
    output

    label_idx
    tp
    ... examples
    fp_misclassification
    ... examples
    fp_hallucination
    ... examples
    fn_misclassification
    ... examples
    fn_missing_prediction
    ... examples
    """

    n_samples = 2

    results = compute_counts_with_examples(
        data=sorted_pairs,
        label_metadata=label_metadata,
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
        n_samples=n_samples,
    )

    assert len(results) == 1
    assert results.shape == (1, 100, 2, 25)  # iou, score, label, metrics

    tp_idx = 0
    fp_misclf_idx = 2 * n_samples + 1
    fp_halluc_idx = 4 * n_samples + 2
    fn_misclf_idx = 6 * n_samples + 3
    fn_misprd_idx = 8 * n_samples + 4

    metric_indices = np.zeros((25,), dtype=bool)
    for index in [
        tp_idx,
        fp_misclf_idx,
        fp_halluc_idx,
        fn_misclf_idx,
        fn_misprd_idx,
    ]:
        metric_indices[index] = True

    """
    @ iou=0.5, score<0.1
    3x tp
    1x fp misclassification
    1x fp hallucination
    0x fn misclassification
    1x fn missing prediction
    """

    assert np.isclose(
        results[0, :10, 0, metric_indices],
        np.array([3, 1, 1, 0, 1])[:, np.newaxis],
    ).all()  # metrics
    assert np.isclose(
        results[0, :10, 0, tp_idx + 1 : fp_misclf_idx], np.array([0, 1, 1, 3])
    ).all()  # tp
    assert np.isclose(
        results[0, :10, 0, fp_misclf_idx + 1 : fp_halluc_idx],
        np.array([1, 2, -1, -1]),
    ).all()  # fp misclf
    assert np.isclose(
        results[0, :10, 0, fp_halluc_idx + 1 : fn_misclf_idx],
        np.array([2, 4, -1, -1]),
    ).all()  # fp halluc
    assert np.isclose(
        results[0, :10, 0, fn_misclf_idx + 1 : fn_misprd_idx],
        np.array([-1, -1, -1, -1]),
    ).all()  # fn misclf
    assert np.isclose(
        results[0, :10, 0, fn_misprd_idx + 1 :], np.array([4, 5, -1, -1])
    ).all()  # fn misprd

    """
    @ iou=0.5, 0.1 <= score < 0.65
    1x tp
    1x fp misclassification
    1x fp hallucination
    1x fn misclassification
    2x fn missing prediction
    """
    assert np.isclose(
        results[0, 10:65, 0, metric_indices],
        np.array([1, 1, 1, 1, 2])[:, np.newaxis],
    ).all()
    assert np.isclose(
        results[0, 10:65, 0, tp_idx + 1 : fp_misclf_idx],
        np.array([0, 1, -1, -1]),
    ).all()  # tp
    assert np.isclose(
        results[0, 10:65, 0, fp_misclf_idx + 1 : fp_halluc_idx],
        np.array([1, 2, -1, -1]),
    ).all()  # fp misclf
    assert np.isclose(
        results[0, 10:65, 0, fp_halluc_idx + 1 : fn_misclf_idx],
        np.array([2, 4, -1, -1]),
    ).all()  # fp halluc
    assert np.isclose(
        results[0, 10:65, 0, fn_misclf_idx + 1 : fn_misprd_idx],
        np.array([1, 1, -1, -1]),
    ).all()  # fn misclf
    assert np.isclose(
        results[0, 10:65, 0, fn_misprd_idx + 1 :], np.array([3, 4, 4, 5])
    ).all()  # fn misprd

    """
    @ iou=0.5, 0.65 <= score < 0.9
    1x tp
    1x fp misclassification
    0x fp hallucination
    1x fn misclassification
    2x fn missing prediction
    """
    assert np.isclose(
        results[0, 65:90, 0, metric_indices],
        np.array([1, 1, 0, 1, 2])[:, np.newaxis],
    ).all()
    assert np.isclose(
        results[0, 65:90, 0, tp_idx + 1 : fp_misclf_idx],
        np.array([0, 1, -1, -1]),
    ).all()  # tp
    assert np.isclose(
        results[0, 65:90, 0, fp_misclf_idx + 1 : fp_halluc_idx],
        np.array([1, 2, -1, -1]),
    ).all()  # fp misclf
    assert np.isclose(
        results[0, 65:90, 0, fp_halluc_idx + 1 : fn_misclf_idx],
        np.array([-1, -1, -1, -1]),
    ).all()  # fp halluc
    assert np.isclose(
        results[0, 65:90, 0, fn_misclf_idx + 1 : fn_misprd_idx],
        np.array([1, 1, -1, -1]),
    ).all()  # fn misclf
    assert np.isclose(
        results[0, 65:90, 0, fn_misprd_idx + 1 :], np.array([3, 4, 4, 5])
    ).all()  # fn misprd

    """
    @ iou=0.5, score>=0.9
    0x tp
    0x fp misclassification
    0x fp hallucination
    0x fn misclassification
    4x fn missing prediction
    """
    assert np.isclose(
        results[0, 95:, 0, metric_indices],
        np.array([0, 0, 0, 0, 4])[:, np.newaxis],
    ).all()
    assert np.isclose(
        results[0, 95:, 0, tp_idx + 1 : fp_misclf_idx],
        np.array([-1, -1, -1, -1]),
    ).all()  # tp
    assert np.isclose(
        results[0, 95:, 0, fp_misclf_idx + 1 : fp_halluc_idx],
        np.array([-1, -1, -1, -1]),
    ).all()  # fp misclf
    assert np.isclose(
        results[0, 95:, 0, fp_halluc_idx + 1 : fn_misclf_idx],
        np.array([-1, -1, -1, -1]),
    ).all()  # fp halluc
    assert np.isclose(
        results[0, 95:, 0, fn_misclf_idx + 1 : fn_misprd_idx],
        np.array([-1, -1, -1, -1]),
    ).all()  # fn misclf
    assert np.isclose(
        results[0, 95:, 0, fn_misprd_idx + 1 :], np.array([0, 0, 1, 1])
    ).all()  # fn misprd


def test_counts_with_examples(
    detections_for_detailed_counting: list[Detection],
    rect1: tuple[float, float, float, float],
    rect2: tuple[float, float, float, float],
    rect3: tuple[float, float, float, float],
    rect4: tuple[float, float, float, float],
    rect5: tuple[float, float, float, float],
):
    loader = DataLoader()
    loader.add_bounding_boxes(detections_for_detailed_counting)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == [
        ("k1", "not_v2"),
        ("k1", "hallucination"),
    ]
    assert evaluator.missing_prediction_labels == [
        ("k1", "missed_detection"),
        ("k1", "v2"),
    ]
    assert evaluator.n_datums == 2
    assert evaluator.n_labels == 6
    assert evaluator.n_groundtruths == 4
    assert evaluator.n_predictions == 4

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        score_thresholds=[0.05, 0.3, 0.35, 0.45, 0.55, 0.95],
        number_of_examples=1,
        metrics_to_return=[MetricType.DetailedCounts],
    )

    uid1_rect1 = ("uid1", rect1)
    uid1_rect2 = ("uid1", rect2)
    uid1_rect3 = ("uid1", rect3)
    uid1_rect4 = ("uid1", rect4)
    uid1_rect5 = ("uid1", rect5)
    uid2_rect1 = ("uid2", rect1)
    uid2_rect2 = ("uid2", rect2)

    # test DetailedCounts
    actual_metrics = [m.to_dict() for m in metrics[MetricType.DetailedCounts]]
    expected_metrics = [
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [1, 1, 1, 1, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0],
                "fp_hallucination": [0, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [0, 0, 0, 0, 1, 1],
                "tp_examples": [
                    [uid1_rect1],
                    [uid1_rect1],
                    [uid1_rect1],
                    [uid1_rect1],
                    [],
                    [],
                ],
                "fp_misclassification_examples": [[], [], [], [], [], []],
                "fp_hallucination_examples": [[], [], [], [], [], []],
                "fn_misclassification_examples": [[], [], [], [], [], []],
                "fn_missing_prediction_examples": [
                    [],
                    [],
                    [],
                    [],
                    [uid1_rect1],
                    [uid1_rect1],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.3,
                    0.35,
                    0.45,
                    0.55,
                    0.95,
                ],
                "iou_threshold": 0.5,
                "label": {"key": "k1", "value": "v1"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [0, 0, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0],
                "fp_hallucination": [0, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [1, 1, 1, 1, 1, 1],
                "tp_examples": [[], [], [], [], [], []],
                "fp_misclassification_examples": [[], [], [], [], [], []],
                "fp_hallucination_examples": [[], [], [], [], [], []],
                "fn_misclassification_examples": [[], [], [], [], [], []],
                "fn_missing_prediction_examples": [
                    [uid1_rect2],
                    [uid1_rect2],
                    [uid1_rect2],
                    [uid1_rect2],
                    [uid1_rect2],
                    [uid1_rect2],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.3,
                    0.35,
                    0.45,
                    0.55,
                    0.95,
                ],
                "iou_threshold": 0.5,
                "label": {"key": "k1", "value": "missed_detection"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [0, 0, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0],
                "fp_hallucination": [0, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [1, 1, 1, 1, 1, 1],
                "tp_examples": [[], [], [], [], [], []],
                "fp_misclassification_examples": [[], [], [], [], [], []],
                "fp_hallucination_examples": [[], [], [], [], [], []],
                "fn_misclassification_examples": [[], [], [], [], [], []],
                "fn_missing_prediction_examples": [
                    [uid1_rect3],
                    [uid1_rect3],
                    [uid1_rect3],
                    [uid1_rect3],
                    [uid1_rect3],
                    [uid1_rect3],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.3,
                    0.35,
                    0.45,
                    0.55,
                    0.95,
                ],
                "iou_threshold": 0.5,
                "label": {"key": "k1", "value": "v2"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [0, 0, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0],
                "fp_hallucination": [1, 1, 1, 1, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [1, 1, 1, 1, 1, 1],
                "tp_examples": [[], [], [], [], [], []],
                "fp_misclassification_examples": [[], [], [], [], [], []],
                "fp_hallucination_examples": [
                    [uid2_rect2],
                    [uid2_rect2],
                    [uid2_rect2],
                    [uid2_rect2],
                    [],
                    [],
                ],
                "fn_misclassification_examples": [[], [], [], [], [], []],
                "fn_missing_prediction_examples": [
                    [uid2_rect1],
                    [uid2_rect1],
                    [uid2_rect1],
                    [uid2_rect1],
                    [uid2_rect1],
                    [uid2_rect1],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.3,
                    0.35,
                    0.45,
                    0.55,
                    0.95,
                ],
                "iou_threshold": 0.5,
                "label": {"key": "k1", "value": "low_iou"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [0, 0, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0],
                "fp_hallucination": [1, 1, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [0, 0, 0, 0, 0, 0],
                "tp_examples": [[], [], [], [], [], []],
                "fp_misclassification_examples": [[], [], [], [], [], []],
                "fp_hallucination_examples": [
                    [uid1_rect5],
                    [uid1_rect5],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_misclassification_examples": [[], [], [], [], [], []],
                "fn_missing_prediction_examples": [[], [], [], [], [], []],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.3,
                    0.35,
                    0.45,
                    0.55,
                    0.95,
                ],
                "iou_threshold": 0.5,
                "label": {"key": "k1", "value": "not_v2"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [0, 0, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0],
                "fp_hallucination": [1, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [0, 0, 0, 0, 0, 0],
                "tp_examples": [[], [], [], [], [], []],
                "fp_misclassification_examples": [[], [], [], [], [], []],
                "fp_hallucination_examples": [
                    [uid1_rect4],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_misclassification_examples": [[], [], [], [], [], []],
                "fn_missing_prediction_examples": [[], [], [], [], [], []],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.3,
                    0.35,
                    0.45,
                    0.55,
                    0.95,
                ],
                "iou_threshold": 0.5,
                "label": {"key": "k1", "value": "hallucination"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test at lower IoU threshold

    metrics = evaluator.evaluate(
        iou_thresholds=[0.45],
        score_thresholds=[0.05, 0.3, 0.35, 0.45, 0.55, 0.95],
        number_of_examples=1,
        metrics_to_return=[MetricType.DetailedCounts],
    )

    # test DetailedCounts
    actual_metrics = [m.to_dict() for m in metrics[MetricType.DetailedCounts]]
    expected_metrics = [
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [1, 1, 1, 1, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0],
                "fp_hallucination": [0, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [0, 0, 0, 0, 1, 1],
                "tp_examples": [
                    [uid1_rect1],
                    [uid1_rect1],
                    [uid1_rect1],
                    [uid1_rect1],
                    [],
                    [],
                ],
                "fp_misclassification_examples": [[], [], [], [], [], []],
                "fp_hallucination_examples": [[], [], [], [], [], []],
                "fn_misclassification_examples": [[], [], [], [], [], []],
                "fn_missing_prediction_examples": [
                    [],
                    [],
                    [],
                    [],
                    [uid1_rect1],
                    [uid1_rect1],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.3,
                    0.35,
                    0.45,
                    0.55,
                    0.95,
                ],
                "iou_threshold": 0.45,
                "label": {"key": "k1", "value": "v1"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [0, 0, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0],
                "fp_hallucination": [0, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [1, 1, 1, 1, 1, 1],
                "tp_examples": [[], [], [], [], [], []],
                "fp_misclassification_examples": [[], [], [], [], [], []],
                "fp_hallucination_examples": [[], [], [], [], [], []],
                "fn_misclassification_examples": [[], [], [], [], [], []],
                "fn_missing_prediction_examples": [
                    [uid1_rect2],
                    [uid1_rect2],
                    [uid1_rect2],
                    [uid1_rect2],
                    [uid1_rect2],
                    [uid1_rect2],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.3,
                    0.35,
                    0.45,
                    0.55,
                    0.95,
                ],
                "iou_threshold": 0.45,
                "label": {"key": "k1", "value": "missed_detection"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [0, 0, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0],
                "fp_hallucination": [0, 0, 0, 0, 0, 0],
                "fn_misclassification": [1, 1, 0, 0, 0, 0],
                "fn_missing_prediction": [0, 0, 1, 1, 1, 1],
                "tp_examples": [[], [], [], [], [], []],
                "fp_misclassification_examples": [[], [], [], [], [], []],
                "fp_hallucination_examples": [[], [], [], [], [], []],
                "fn_misclassification_examples": [
                    [uid1_rect3],
                    [uid1_rect3],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_missing_prediction_examples": [
                    [],
                    [],
                    [uid1_rect3],
                    [uid1_rect3],
                    [uid1_rect3],
                    [uid1_rect3],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.3,
                    0.35,
                    0.45,
                    0.55,
                    0.95,
                ],
                "iou_threshold": 0.45,
                "label": {"key": "k1", "value": "v2"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [0, 0, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0],
                "fp_hallucination": [1, 1, 1, 1, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [1, 1, 1, 1, 1, 1],
                "tp_examples": [[], [], [], [], [], []],
                "fp_misclassification_examples": [[], [], [], [], [], []],
                "fp_hallucination_examples": [
                    [uid2_rect2],
                    [uid2_rect2],
                    [uid2_rect2],
                    [uid2_rect2],
                    [],
                    [],
                ],
                "fn_misclassification_examples": [[], [], [], [], [], []],
                "fn_missing_prediction_examples": [
                    [uid2_rect1],
                    [uid2_rect1],
                    [uid2_rect1],
                    [uid2_rect1],
                    [uid2_rect1],
                    [uid2_rect1],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.3,
                    0.35,
                    0.45,
                    0.55,
                    0.95,
                ],
                "iou_threshold": 0.45,
                "label": {"key": "k1", "value": "low_iou"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [0, 0, 0, 0, 0, 0],
                "fp_misclassification": [1, 1, 0, 0, 0, 0],
                "fp_hallucination": [0, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [0, 0, 0, 0, 0, 0],
                "tp_examples": [[], [], [], [], [], []],
                "fp_misclassification_examples": [
                    [uid1_rect5],
                    [uid1_rect5],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_hallucination_examples": [[], [], [], [], [], []],
                "fn_misclassification_examples": [[], [], [], [], [], []],
                "fn_missing_prediction_examples": [[], [], [], [], [], []],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.3,
                    0.35,
                    0.45,
                    0.55,
                    0.95,
                ],
                "iou_threshold": 0.45,
                "label": {"key": "k1", "value": "not_v2"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [0, 0, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0],
                "fp_hallucination": [1, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [0, 0, 0, 0, 0, 0],
                "tp_examples": [[], [], [], [], [], []],
                "fp_misclassification_examples": [[], [], [], [], [], []],
                "fp_hallucination_examples": [
                    [uid1_rect4],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_misclassification_examples": [[], [], [], [], [], []],
                "fn_missing_prediction_examples": [[], [], [], [], [], []],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.3,
                    0.35,
                    0.45,
                    0.55,
                    0.95,
                ],
                "iou_threshold": 0.45,
                "label": {"key": "k1", "value": "hallucination"},
            },
        },
    ]

    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_with_examples_using_torch_metrics_example(
    torchmetrics_detections: list[Detection],
):
    """
    cf with torch metrics/pycocotools results listed here:
    https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    """
    loader = DataLoader()
    loader.add_bounding_boxes(torchmetrics_detections)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == [("class", "3")]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 4
    assert evaluator.n_labels == 6
    assert evaluator.n_groundtruths == 20
    assert evaluator.n_predictions == 19

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5, 0.9],
        score_thresholds=[0.05, 0.25, 0.35, 0.55, 0.75, 0.8, 0.85, 0.95],
        number_of_examples=1,
        metrics_to_return=[MetricType.DetailedCounts],
    )

    assert len(metrics[MetricType.DetailedCounts]) == 12

    uid0_gt_0 = ("0", (214.125, 562.5, 41.28125, 285.0))
    uid1_gt_0 = ("1", (13.0, 549.0, 22.75, 632.5))
    uid2_gt_1 = ("2", (2.75, 162.125, 3.66015625, 316.0))
    uid2_gt_2 = ("2", (295.5, 314.0, 93.9375, 152.75))
    uid2_gt_4 = ("2", (356.5, 372.25, 95.5, 147.5))
    uid3_gt_0 = ("3", (72.9375, 91.25, 45.96875, 80.5625))
    uid3_gt_1 = ("3", (50.15625, 71.25, 45.34375, 79.8125))
    uid3_gt_5 = ("3", (56.375, 75.6875, 21.65625, 45.53125))

    uid0_pd_0 = ("0", (258.25, 606.5, 41.28125, 285.0))
    uid1_pd_0 = ("1", (61.0, 565.0, 22.75, 632.5))
    uid1_pd_1 = ("1", (12.65625, 281.25, 3.3203125, 275.25))
    uid2_pd_0 = ("2", (87.875, 384.25, 276.25, 379.5))
    uid2_pd_1 = ("2", (0.0, 142.125, 3.66015625, 316.0))
    uid2_pd_2 = ("2", (296.5, 315.0, 93.9375, 152.75))
    uid2_pd_3 = ("2", (329.0, 342.5, 97.0625, 123.0))
    uid2_pd_4 = ("2", (356.5, 372.25, 95.5, 147.5))
    uid2_pd_5 = ("2", (464.0, 495.75, 105.0625, 147.0))
    uid2_pd_6 = ("2", (276.0, 291.5, 103.8125, 150.75))
    uid3_pd_0 = ("3", (72.9375, 91.25, 45.96875, 80.5625))
    uid3_pd_1 = ("3", (45.15625, 66.25, 45.34375, 79.8125))
    uid3_pd_2 = ("3", (82.25, 99.6875, 47.03125, 78.5))
    uid3_pd_4 = ("3", (75.3125, 91.875, 23.015625, 50.84375))

    # test DetailedCounts
    actual_metrics = [m.to_dict() for m in metrics[MetricType.DetailedCounts]]
    expected_metrics = [
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [2, 1, 1, 0, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fp_hallucination": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [0, 1, 1, 2, 2, 2, 2, 2],
                "tp_examples": [
                    [uid0_pd_0],
                    [uid2_pd_0],
                    [uid2_pd_0],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_hallucination_examples": [[], [], [], [], [], [], [], []],
                "fn_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_missing_prediction_examples": [
                    [],
                    [uid0_gt_0],
                    [uid0_gt_0],
                    [uid0_gt_0],
                    [uid0_gt_0],
                    [uid0_gt_0],
                    [uid0_gt_0],
                    [uid0_gt_0],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.25,
                    0.35,
                    0.55,
                    0.75,
                    0.8,
                    0.85,
                    0.95,
                ],
                "iou_threshold": 0.5,
                "label": {"key": "class", "value": "4"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [0, 0, 0, 0, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fp_hallucination": [2, 1, 1, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [2, 2, 2, 2, 2, 2, 2, 2],
                "tp_examples": [[], [], [], [], [], [], [], []],
                "fp_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_hallucination_examples": [
                    [uid0_pd_0],
                    [uid2_pd_0],
                    [uid2_pd_0],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_missing_prediction_examples": [
                    [uid0_gt_0],
                    [uid0_gt_0],
                    [uid0_gt_0],
                    [uid0_gt_0],
                    [uid0_gt_0],
                    [uid0_gt_0],
                    [uid0_gt_0],
                    [uid0_gt_0],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.25,
                    0.35,
                    0.55,
                    0.75,
                    0.8,
                    0.85,
                    0.95,
                ],
                "iou_threshold": 0.9,
                "label": {"key": "class", "value": "4"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [1, 1, 1, 1, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fp_hallucination": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_misclassification": [1, 1, 0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [0, 0, 1, 1, 2, 2, 2, 2],
                "tp_examples": [
                    [uid1_pd_1],
                    [uid1_pd_1],
                    [uid1_pd_1],
                    [uid1_pd_1],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_hallucination_examples": [[], [], [], [], [], [], [], []],
                "fn_misclassification_examples": [
                    [uid1_gt_0],
                    [uid1_gt_0],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_missing_prediction_examples": [
                    [],
                    [],
                    [uid1_gt_0],
                    [uid1_gt_0],
                    [uid1_gt_0],
                    [uid1_gt_0],
                    [uid1_gt_0],
                    [uid1_gt_0],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.25,
                    0.35,
                    0.55,
                    0.75,
                    0.8,
                    0.85,
                    0.95,
                ],
                "iou_threshold": 0.5,
                "label": {"key": "class", "value": "2"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [1, 1, 1, 1, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fp_hallucination": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [1, 1, 1, 1, 2, 2, 2, 2],
                "tp_examples": [
                    [uid1_pd_1],
                    [uid1_pd_1],
                    [uid1_pd_1],
                    [uid1_pd_1],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_hallucination_examples": [[], [], [], [], [], [], [], []],
                "fn_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_missing_prediction_examples": [
                    [uid1_gt_0],
                    [uid1_gt_0],
                    [uid1_gt_0],
                    [uid1_gt_0],
                    [uid1_gt_0],
                    [uid1_gt_0],
                    [uid1_gt_0],
                    [uid1_gt_0],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.25,
                    0.35,
                    0.55,
                    0.75,
                    0.8,
                    0.85,
                    0.95,
                ],
                "iou_threshold": 0.9,
                "label": {"key": "class", "value": "2"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [1, 1, 0, 0, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fp_hallucination": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [0, 0, 1, 1, 1, 1, 1, 1],
                "tp_examples": [
                    [uid2_pd_1],
                    [uid2_pd_1],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_hallucination_examples": [[], [], [], [], [], [], [], []],
                "fn_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_missing_prediction_examples": [
                    [],
                    [],
                    [uid2_gt_1],
                    [uid2_gt_1],
                    [uid2_gt_1],
                    [uid2_gt_1],
                    [uid2_gt_1],
                    [uid2_gt_1],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.25,
                    0.35,
                    0.55,
                    0.75,
                    0.8,
                    0.85,
                    0.95,
                ],
                "iou_threshold": 0.5,
                "label": {"key": "class", "value": "1"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [0, 0, 0, 0, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fp_hallucination": [1, 1, 0, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [1, 1, 1, 1, 1, 1, 1, 1],
                "tp_examples": [[], [], [], [], [], [], [], []],
                "fp_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_hallucination_examples": [
                    [uid2_pd_1],
                    [uid2_pd_1],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_missing_prediction_examples": [
                    [uid2_gt_1],
                    [uid2_gt_1],
                    [uid2_gt_1],
                    [uid2_gt_1],
                    [uid2_gt_1],
                    [uid2_gt_1],
                    [uid2_gt_1],
                    [uid2_gt_1],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.25,
                    0.35,
                    0.55,
                    0.75,
                    0.8,
                    0.85,
                    0.95,
                ],
                "iou_threshold": 0.9,
                "label": {"key": "class", "value": "1"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [5, 5, 4, 3, 2, 2, 1, 1],
                "fp_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fp_hallucination": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [0, 0, 1, 2, 3, 3, 4, 4],
                "tp_examples": [
                    [uid2_pd_2],
                    [uid2_pd_2],
                    [uid2_pd_2],
                    [uid2_pd_3],
                    [uid2_pd_5],
                    [uid2_pd_5],
                    [uid2_pd_6],
                    [uid2_pd_6],
                ],
                "fp_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_hallucination_examples": [[], [], [], [], [], [], [], []],
                "fn_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_missing_prediction_examples": [
                    [],
                    [],
                    [uid2_gt_4],
                    [uid2_gt_2],
                    [uid2_gt_2],
                    [uid2_gt_2],
                    [uid2_gt_2],
                    [uid2_gt_2],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.25,
                    0.35,
                    0.55,
                    0.75,
                    0.8,
                    0.85,
                    0.95,
                ],
                "iou_threshold": 0.5,
                "label": {"key": "class", "value": "0"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [1, 1, 0, 0, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fp_hallucination": [4, 4, 4, 3, 2, 2, 1, 1],
                "fn_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [4, 4, 5, 5, 5, 5, 5, 5],
                "tp_examples": [
                    [uid2_pd_4],
                    [uid2_pd_4],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_hallucination_examples": [
                    [uid2_pd_2],
                    [uid2_pd_2],
                    [uid2_pd_2],
                    [uid2_pd_3],
                    [uid2_pd_5],
                    [uid2_pd_5],
                    [uid2_pd_6],
                    [uid2_pd_6],
                ],
                "fn_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_missing_prediction_examples": [
                    [uid2_gt_2],
                    [uid2_gt_2],
                    [uid2_gt_2],
                    [uid2_gt_2],
                    [uid2_gt_2],
                    [uid2_gt_2],
                    [uid2_gt_2],
                    [uid2_gt_2],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.25,
                    0.35,
                    0.55,
                    0.75,
                    0.8,
                    0.85,
                    0.95,
                ],
                "iou_threshold": 0.9,
                "label": {"key": "class", "value": "0"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [9, 6, 4, 3, 2, 1, 1, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fp_hallucination": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [1, 4, 6, 7, 8, 9, 9, 10],
                "tp_examples": [
                    [uid3_pd_0],
                    [uid3_pd_0],
                    [uid3_pd_0],
                    [uid3_pd_2],
                    [uid3_pd_2],
                    [uid3_pd_4],
                    [uid3_pd_4],
                    [],
                ],
                "fp_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_hallucination_examples": [[], [], [], [], [], [], [], []],
                "fn_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_missing_prediction_examples": [
                    [uid3_gt_5],
                    [uid3_gt_1],
                    [uid3_gt_1],
                    [uid3_gt_0],
                    [uid3_gt_0],
                    [uid3_gt_0],
                    [uid3_gt_0],
                    [uid3_gt_0],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.25,
                    0.35,
                    0.55,
                    0.75,
                    0.8,
                    0.85,
                    0.95,
                ],
                "iou_threshold": 0.5,
                "label": {"key": "class", "value": "49"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [2, 2, 2, 1, 1, 1, 1, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fp_hallucination": [7, 4, 2, 2, 1, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [8, 8, 8, 9, 9, 9, 9, 10],
                "tp_examples": [
                    [uid3_pd_0],
                    [uid3_pd_0],
                    [uid3_pd_0],
                    [uid3_pd_4],
                    [uid3_pd_4],
                    [uid3_pd_4],
                    [uid3_pd_4],
                    [],
                ],
                "fp_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_hallucination_examples": [
                    [uid3_pd_1],
                    [uid3_pd_2],
                    [uid3_pd_2],
                    [uid3_pd_2],
                    [uid3_pd_2],
                    [],
                    [],
                    [],
                ],
                "fn_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_missing_prediction_examples": [
                    [uid3_gt_1],
                    [uid3_gt_1],
                    [uid3_gt_1],
                    [uid3_gt_0],
                    [uid3_gt_0],
                    [uid3_gt_0],
                    [uid3_gt_0],
                    [uid3_gt_0],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.25,
                    0.35,
                    0.55,
                    0.75,
                    0.8,
                    0.85,
                    0.95,
                ],
                "iou_threshold": 0.9,
                "label": {"key": "class", "value": "49"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [0, 0, 0, 0, 0, 0, 0, 0],
                "fp_misclassification": [1, 1, 0, 0, 0, 0, 0, 0],
                "fp_hallucination": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [0, 0, 0, 0, 0, 0, 0, 0],
                "tp_examples": [[], [], [], [], [], [], [], []],
                "fp_misclassification_examples": [
                    [uid1_pd_0],
                    [uid1_pd_0],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_hallucination_examples": [[], [], [], [], [], [], [], []],
                "fn_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_missing_prediction_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.25,
                    0.35,
                    0.55,
                    0.75,
                    0.8,
                    0.85,
                    0.95,
                ],
                "iou_threshold": 0.5,
                "label": {"key": "class", "value": "3"},
            },
        },
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [0, 0, 0, 0, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fp_hallucination": [1, 1, 0, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [0, 0, 0, 0, 0, 0, 0, 0],
                "tp_examples": [[], [], [], [], [], [], [], []],
                "fp_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fp_hallucination_examples": [
                    [uid1_pd_0],
                    [uid1_pd_0],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_misclassification_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_missing_prediction_examples": [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                ],
            },
            "parameters": {
                "score_thresholds": [
                    0.05,
                    0.25,
                    0.35,
                    0.55,
                    0.75,
                    0.8,
                    0.85,
                    0.95,
                ],
                "iou_threshold": 0.9,
                "label": {"key": "class", "value": "3"},
            },
        },
    ]

    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_with_examples_fp_hallucination_edge_case(
    detections_fp_hallucination_edge_case: list[Detection],
):

    loader = DataLoader()
    loader.add_bounding_boxes(detections_fp_hallucination_edge_case)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 2
    assert evaluator.n_labels == 1
    assert evaluator.n_groundtruths == 2
    assert evaluator.n_predictions == 2

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        score_thresholds=[0.5, 0.85],
        number_of_examples=1,
        metrics_to_return=[MetricType.DetailedCounts],
    )

    assert len(metrics[MetricType.DetailedCounts]) == 1

    # test DetailedCounts
    actual_metrics = [m.to_dict() for m in metrics[MetricType.DetailedCounts]]
    expected_metrics = [
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [1, 0],
                "fp_misclassification": [0, 0],
                "fp_hallucination": [1, 0],
                "fn_misclassification": [0, 0],
                "fn_missing_prediction": [1, 2],
                "tp_examples": [[("uid1", (0.0, 5.0, 0.0, 5.0))], []],
                "fp_misclassification_examples": [[], []],
                "fp_hallucination_examples": [
                    [("uid2", (10.0, 20.0, 10.0, 20.0))],
                    [],
                ],
                "fn_misclassification_examples": [[], []],
                "fn_missing_prediction_examples": [
                    [("uid2", (0.0, 5.0, 0.0, 5.0))],
                    [("uid1", (0.0, 5.0, 0.0, 5.0))],
                ],
            },
            "parameters": {
                "score_thresholds": [0.5, 0.85],
                "iou_threshold": 0.5,
                "label": {"key": "k1", "value": "v1"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_with_examples_ranked_pair_ordering(
    detection_ranked_pair_ordering: Detection,
    detection_ranked_pair_ordering_with_bitmasks: Detection,
    detection_ranked_pair_ordering_with_polygons: Detection,
):

    for input_, method in [
        (detection_ranked_pair_ordering, DataLoader.add_bounding_boxes),
        (
            detection_ranked_pair_ordering_with_bitmasks,
            DataLoader.add_bitmasks,
        ),
        (
            detection_ranked_pair_ordering_with_polygons,
            DataLoader.add_polygons,
        ),
    ]:
        loader = DataLoader()
        method(loader, detections=[input_])

        evaluator = loader.finalize()

        assert evaluator.metadata == {
            "ignored_prediction_labels": [
                ("class", "label4"),
            ],
            "missing_prediction_labels": [],
            "n_datums": 1,
            "n_groundtruths": 3,
            "n_labels": 4,
            "n_predictions": 4,
        }

        metrics = evaluator.evaluate(
            iou_thresholds=[0.5],
            score_thresholds=[0.0],
            number_of_examples=0,
            metrics_to_return=[MetricType.DetailedCounts],
        )

        actual_metrics = [
            m.to_dict() for m in metrics[MetricType.DetailedCounts]
        ]
        expected_metrics = [
            {
                "type": "DetailedCounts",
                "value": {
                    "tp": [1],
                    "fp_misclassification": [0],
                    "fp_hallucination": [0],
                    "fn_misclassification": [0],
                    "fn_missing_prediction": [0],
                    "tp_examples": [[]],
                    "fp_misclassification_examples": [[]],
                    "fp_hallucination_examples": [[]],
                    "fn_misclassification_examples": [[]],
                    "fn_missing_prediction_examples": [[]],
                },
                "parameters": {
                    "score_thresholds": [0.0],
                    "iou_threshold": 0.5,
                    "label": {"key": "class", "value": "label1"},
                },
            },
            {
                "type": "DetailedCounts",
                "value": {
                    "tp": [1],
                    "fp_misclassification": [0],
                    "fp_hallucination": [0],
                    "fn_misclassification": [0],
                    "fn_missing_prediction": [0],
                    "tp_examples": [[]],
                    "fp_misclassification_examples": [[]],
                    "fp_hallucination_examples": [[]],
                    "fn_misclassification_examples": [[]],
                    "fn_missing_prediction_examples": [[]],
                },
                "parameters": {
                    "score_thresholds": [0.0],
                    "iou_threshold": 0.5,
                    "label": {"key": "class", "value": "label2"},
                },
            },
            {
                "type": "DetailedCounts",
                "value": {
                    "tp": [0],
                    "fp_misclassification": [0],
                    "fp_hallucination": [1],
                    "fn_misclassification": [1],
                    "fn_missing_prediction": [0],
                    "tp_examples": [[]],
                    "fp_misclassification_examples": [[]],
                    "fp_hallucination_examples": [[]],
                    "fn_misclassification_examples": [[]],
                    "fn_missing_prediction_examples": [[]],
                },
                "parameters": {
                    "score_thresholds": [0.0],
                    "iou_threshold": 0.5,
                    "label": {"key": "class", "value": "label3"},
                },
            },
            {
                "type": "DetailedCounts",
                "value": {
                    "tp": [0],
                    "fp_misclassification": [0],
                    "fp_hallucination": [1],
                    "fn_misclassification": [0],
                    "fn_missing_prediction": [0],
                    "tp_examples": [[]],
                    "fp_misclassification_examples": [[]],
                    "fp_hallucination_examples": [[]],
                    "fn_misclassification_examples": [[]],
                    "fn_missing_prediction_examples": [[]],
                },
                "parameters": {
                    "score_thresholds": [0.0],
                    "iou_threshold": 0.5,
                    "label": {"key": "class", "value": "label4"},
                },
            },
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics
