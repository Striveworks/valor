from valor_lite.semantic_segmentation import Loader, MetricType, Segmentation


def test_f1_basic_segmentations(
    loader: Loader,
    basic_segmentations: list[Segmentation],
):
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()

    metrics = evaluator.compute_precision_recall_iou()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.F1]]
    expected_metrics = [
        {
            "type": "F1",
            "value": 2 / 3,
            "parameters": {"label": "v1"},
        },
        {
            "type": "F1",
            "value": 2 / 3,
            "parameters": {"label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_f1_segmentations_from_boxes(
    loader: Loader,
    segmentations_from_boxes: list[Segmentation],
):
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    metrics = evaluator.compute_precision_recall_iou()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.F1]]
    expected_metrics = [
        {
            "type": "F1",
            "value": 0.5,  # 50% overlap
            "parameters": {"label": "v1"},
        },
        {
            "type": "F1",
            "value": 1 / 10000,  # overlaps 1 pixel out of 20,000
            "parameters": {"label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
