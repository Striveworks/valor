from valor_lite.semantic_segmentation import (
    IOU,
    DataLoader,
    MetricType,
    Segmentation,
    mIOU,
)


def test_iou_basic_segmentations(basic_segmentations: list[Segmentation]):
    loader = DataLoader()
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.IOU]]
    expected_metrics = [
        {
            "type": "IOU",
            "value": 0.5,
            "parameters": {"label": "v1"},
        },
        {
            "type": "IOU",
            "value": 0.5,
            "parameters": {"label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    actual_metrics = [m for m in metrics[MetricType.mIOU]]
    expected_metrics = [
        {
            "type": "mIOU",
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

    actual_metrics = [m for m in metrics[MetricType.IOU]]
    expected_metrics = [
        {
            "type": "IOU",
            "value": 1 / 3,  # 50% overlap
            "parameters": {"label": "v1"},
        },
        {
            "type": "IOU",
            "value": 1 / 19999,  # overlaps 1 pixel out of 20,000
            "parameters": {"label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    actual_metrics = [m for m in metrics[MetricType.mIOU]]
    expected_metrics = [
        {
            "type": "mIOU",
            "value": ((1 / 3) + (1 / 19999)) / 2,
            "parameters": {},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_recall_large_random_segmentations(
    large_random_segmentations: list[Segmentation],
):
    loader = DataLoader()
    loader.add_data(large_random_segmentations)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate()

    for m in metrics[MetricType.IOU]:
        assert isinstance(m, IOU)
        match m.label:
            case "v1":
                assert round(m.value, 2) == 0.82
            case "v2":
                assert round(m.value, 2) == 0.05
            case "v3":
                assert round(m.value, 1) == 0.0
            case "v4":
                assert round(m.value, 2) == 0.25
            case "v5":
                assert round(m.value, 2) == 0.25
            case "v6":
                assert round(m.value, 2) == 0.05
            case "v7":
                assert round(m.value, 2) == 0.18
            case "v8":
                assert round(m.value, 2) == 0.18
            case "v9":
                assert round(m.value, 2) == 0.18
            case _:
                assert False

    mIOUs = metrics[MetricType.mIOU]
    assert len(mIOUs) == 1
    assert isinstance(mIOUs[0], mIOU)
    assert round(mIOUs[0].value, 2) == 0.22
