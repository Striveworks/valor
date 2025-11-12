from valor_lite.semantic_segmentation import Loader, MetricType, Segmentation


def test_recall_basic_segmentations(
    loader: Loader,
    basic_segmentations: list[Segmentation],
):
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()

    metrics = evaluator.compute_precision_recall_iou()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": 0.5,
            "parameters": {"label": "v1"},
        },
        {
            "type": "Recall",
            "value": 1.0,
            "parameters": {"label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_recall_segmentations_from_boxes(
    loader: Loader,
    segmentations_from_boxes: list[Segmentation],
):
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    metrics = evaluator.compute_precision_recall_iou()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": 0.5,  # 50% overlap
            "parameters": {"label": "v1"},
        },
        {
            "type": "Recall",
            "value": 1 / 15000,  # overlaps 1 pixel out of 15,000 groundtruths
            "parameters": {"label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
