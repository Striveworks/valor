from pathlib import Path

from valor_lite.semantic_segmentation import (
    DataLoader,
    MetricType,
    Segmentation,
)


def test_iou_basic_segmentations(
    tmp_path: Path,
    basic_segmentations: list[Segmentation],
):
    loader = DataLoader.create(tmp_path)
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.IOU]]
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

    actual_metrics = [m.to_dict() for m in metrics[MetricType.mIOU]]
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
    tmp_path: Path,
    segmentations_from_boxes: list[Segmentation],
):
    loader = DataLoader.create(tmp_path)
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.IOU]]
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

    actual_metrics = [m.to_dict() for m in metrics[MetricType.mIOU]]
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
