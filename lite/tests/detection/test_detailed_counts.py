import numpy as np
from valor_lite.detection import DataLoader, Detection, Evaluator
from valor_lite.detection.computation import compute_detailed_counts


def test_detailed_counts_no_data():
    evaluator = Evaluator()
    curves = evaluator.compute_detailed_counts()
    assert isinstance(curves, list)
    assert len(curves) == 0


def test_compute_detailed_counts():
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

    results = compute_detailed_counts(
        data=sorted_pairs,
        label_metadata=label_metadata,
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
        n_samples=0,
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

    results = compute_detailed_counts(
        data=sorted_pairs,
        label_metadata=label_metadata,
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
        n_samples=n_samples,
    )

    assert len(results) == 1
    assert results.shape == (1, 100, 2, 15)  # iou, score, label, metrics

    tp_idx = 0
    fp_misclf_idx = tp_idx + n_samples + 1
    fp_halluc_idx = fp_misclf_idx + n_samples + 1
    fn_misclf_idx = fp_halluc_idx + n_samples + 1
    fn_misprd_idx = fn_misclf_idx + n_samples + 1

    metric_indices = np.zeros((15,), dtype=bool)
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
        results[0, :10, 0, tp_idx + 1 : fp_misclf_idx], np.array([0.0, 1.0])
    ).all()  # tp
    assert np.isclose(
        results[0, :10, 0, fp_misclf_idx + 1 : fp_halluc_idx],
        np.array([1.0, -1.0]),
    ).all()  # fp misclf
    assert np.isclose(
        results[0, :10, 0, fp_halluc_idx + 1 : fn_misclf_idx],
        np.array([2.0, -1.0]),
    ).all()  # fp halluc
    assert np.isclose(
        results[0, :10, 0, fn_misclf_idx + 1 : fn_misprd_idx],
        np.array([-1.0, -1.0]),
    ).all()  # fn misclf
    assert np.isclose(
        results[0, :10, 0, fn_misprd_idx + 1 :], np.array([4.0, -1.0])
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
        results[0, 10:65, 0, tp_idx + 1 : fp_misclf_idx], np.array([0.0, -1.0])
    ).all()  # tp
    assert np.isclose(
        results[0, 10:65, 0, fp_misclf_idx + 1 : fp_halluc_idx],
        np.array([1.0, -1.0]),
    ).all()  # fp misclf
    assert np.isclose(
        results[0, 10:65, 0, fp_halluc_idx + 1 : fn_misclf_idx],
        np.array([2.0, -1.0]),
    ).all()  # fp halluc
    assert np.isclose(
        results[0, 10:65, 0, fn_misclf_idx + 1 : fn_misprd_idx],
        np.array([1.0, -1.0]),
    ).all()  # fn misclf
    assert np.isclose(
        results[0, 10:65, 0, fn_misprd_idx + 1 :], np.array([3.0, 4.0])
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
        results[0, 65:90, 0, tp_idx + 1 : fp_misclf_idx], np.array([0.0, -1.0])
    ).all()  # tp
    assert np.isclose(
        results[0, 65:90, 0, fp_misclf_idx + 1 : fp_halluc_idx],
        np.array([1.0, -1.0]),
    ).all()  # fp misclf
    assert np.isclose(
        results[0, 65:90, 0, fp_halluc_idx + 1 : fn_misclf_idx],
        np.array([-1.0, -1.0]),
    ).all()  # fp halluc
    assert np.isclose(
        results[0, 65:90, 0, fn_misclf_idx + 1 : fn_misprd_idx],
        np.array([1.0, -1.0]),
    ).all()  # fn misclf
    assert np.isclose(
        results[0, 65:90, 0, fn_misprd_idx + 1 :], np.array([3.0, 4.0])
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
        results[0, 95:, 0, tp_idx + 1 : fp_misclf_idx], np.array([-1.0, -1.0])
    ).all()  # tp
    assert np.isclose(
        results[0, 95:, 0, fp_misclf_idx + 1 : fp_halluc_idx],
        np.array([-1.0, -1.0]),
    ).all()  # fp misclf
    assert np.isclose(
        results[0, 95:, 0, fp_halluc_idx + 1 : fn_misclf_idx],
        np.array([-1.0, -1.0]),
    ).all()  # fp halluc
    assert np.isclose(
        results[0, 95:, 0, fn_misclf_idx + 1 : fn_misprd_idx],
        np.array([-1.0, -1.0]),
    ).all()  # fn misclf
    assert np.isclose(
        results[0, 95:, 0, fn_misprd_idx + 1 :], np.array([0.0, 1.0])
    ).all()  # fn misprd


def test_detailed_counts_using_torch_metrics_example(
    torchmetrics_detections: list[Detection],
):
    """
    cf with torch metrics/pycocotools results listed here:
    https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    """
    manager = DataLoader()
    manager.add_data(torchmetrics_detections)
    evaluator = manager.finalize()

    print(evaluator.label_to_index[("class", "49")])
    assert evaluator.ignored_prediction_labels == [("class", "3")]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 4
    assert evaluator.n_labels == 6
    assert evaluator.n_groundtruths == 20
    assert evaluator.n_predictions == 19

    metrics = evaluator.compute_detailed_counts(
        iou_thresholds=[0.5, 0.9],
        score_thresholds=[0.05, 0.25, 0.35, 0.55, 0.75, 0.8, 0.85, 0.95],
        n_samples=1,
    )

    assert len(metrics) == 5

    # test DetailedCounts for
    actual_metrics = [mm.to_dict() for m in metrics for mm in m]
    expected_metrics = [
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [2, 1, 1, 0, 0, 0, 0, 0],
                "fp_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fp_hallucination": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_misclassification": [0, 0, 0, 0, 0, 0, 0, 0],
                "fn_missing_prediction": [0, 1, 1, 2, 2, 2, 2, 2],
                "tn": None,
                "tp_examples": [["0"], ["2"], ["2"], [], [], [], [], []],
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
                    ["0"],
                    ["0"],
                    ["0"],
                    ["0"],
                    ["0"],
                    ["0"],
                    ["0"],
                ],
                "tn_examples": None,
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
                "tn": None,
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
                    ["0"],
                    ["2"],
                    ["2"],
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
                    ["0"],
                    ["0"],
                    ["0"],
                    ["0"],
                    ["0"],
                    ["0"],
                    ["0"],
                    ["0"],
                ],
                "tn_examples": None,
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
                "tn": None,
                "tp_examples": [["1"], ["1"], ["1"], ["1"], [], [], [], []],
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
                    ["1"],
                    ["1"],
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
                    ["1"],
                    ["1"],
                    ["1"],
                    ["1"],
                    ["1"],
                    ["1"],
                ],
                "tn_examples": None,
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
                "tn": None,
                "tp_examples": [["1"], ["1"], ["1"], ["1"], [], [], [], []],
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
                    ["1"],
                    ["1"],
                    ["1"],
                    ["1"],
                    ["1"],
                    ["1"],
                    ["1"],
                    ["1"],
                ],
                "tn_examples": None,
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
                "tn": None,
                "tp_examples": [["2"], ["2"], [], [], [], [], [], []],
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
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                ],
                "tn_examples": None,
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
                "tn": None,
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
                    ["2"],
                    ["2"],
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
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                ],
                "tn_examples": None,
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
                "tn": None,
                "tp_examples": [
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
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
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                ],
                "tn_examples": None,
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
                "tn": None,
                "tp_examples": [["2"], ["2"], [], [], [], [], [], []],
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
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
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
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                    ["2"],
                ],
                "tn_examples": None,
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
                "fn_misclassification": [0, 1, 1, 1, 0, 0, 0, 0],
                "fn_missing_prediction": [1, 4, 6, 7, 8, 9, 9, 10],
                "tn": None,
                "tp_examples": [
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
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
                    ["3"],
                    ["3"],
                    ["3"],
                    [],
                    [],
                    [],
                    [],
                ],
                "fn_missing_prediction_examples": [
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
                ],
                "tn_examples": None,
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
                "tn": None,
                "tp_examples": [
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
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
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
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
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
                    ["3"],
                ],
                "tn_examples": None,
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
    ]

    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_detailed_counts_fp_hallucination_edge_case(
    detections_fp_hallucination_edge_case: list[Detection],
):

    manager = DataLoader()
    manager.add_data(detections_fp_hallucination_edge_case)
    evaluator = manager.finalize()

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 2
    assert evaluator.n_labels == 1
    assert evaluator.n_groundtruths == 2
    assert evaluator.n_predictions == 2

    metrics = evaluator.compute_detailed_counts(
        iou_thresholds=[0.5],
        score_thresholds=[0.5, 0.85],
        n_samples=1,
    )

    assert len(metrics) == 1

    # test DetailedCounts
    actual_metrics = [mm.to_dict() for m in metrics for mm in m]
    expected_metrics = [
        {
            "type": "DetailedCounts",
            "value": {
                "tp": [1, 0],
                "fp_misclassification": [0, 0],
                "fp_hallucination": [1, 0],
                "fn_misclassification": [0, 0],
                "fn_missing_prediction": [1, 2],
                "tn": None,
                "tp_examples": [["uid1"], []],
                "fp_misclassification_examples": [[], []],
                "fp_hallucination_examples": [["uid2"], []],
                "fn_misclassification_examples": [[], []],
                "fn_missing_prediction_examples": [["uid2"], ["uid1"]],
                "tn_examples": None,
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
