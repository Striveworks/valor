from valor_lite.segmentation import DataLoader, MetricType, Segmentation


def test_iou_basic_segmenations(basic_segmentations: list[Segmentation]):
    loader = DataLoader()
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.IoU]]
    expected_metrics = [
        {
            "type": "IoU",
            "value": 0.5,
            "parameters": {"label": "v1"},
        },
        {
            "type": "IoU",
            "value": 0.5,
            "parameters": {"label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    actual_metrics = [m for m in metrics[MetricType.mIoU]]
    expected_metrics = [
        {
            "type": "mIoU",
            "value": 0.5,
            "parameters": {},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_iou_segmentations_from_boxes(
    segmentations_from_boxes: list[Segmentation],
):
    loader = DataLoader()
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.IoU]]
    expected_metrics = [
        {
            "type": "IoU",
            "value": 1 / 3,  # 50% overlap
            "parameters": {"label": "v1"},
        },
        {
            "type": "IoU",
            "value": 1 / 19999,  # overlaps 1 pixel out of 20,000
            "parameters": {"label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    actual_metrics = [m for m in metrics[MetricType.mIoU]]
    expected_metrics = [
        {
            "type": "mIoU",
            "value": ((1 / 3) + (1 / 19999)) / 2,
            "parameters": {},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
