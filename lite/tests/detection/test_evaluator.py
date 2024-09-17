from valor_lite.detection import DataLoader, Detection


def test_metadata_using_torch_metrics_example(
    torchmetrics_detections: list[Detection],
):
    """
    cf with torch metrics/pycocotools results listed here:
    https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    """
    manager = DataLoader()
    manager.add_data(torchmetrics_detections)
    evaluator = manager.finalize()

    assert evaluator.ignored_prediction_labels == [("class", "3")]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 4
    assert evaluator.n_labels == 6
    assert evaluator.n_groundtruths == 20
    assert evaluator.n_predictions == 19

    assert evaluator.metadata == {
        "ignored_prediction_labels": [
            ("class", "3"),
        ],
        "missing_prediction_labels": [],
        "n_datums": 4,
        "n_labels": 6,
        "n_groundtruths": 20,
        "n_predictions": 19,
    }
