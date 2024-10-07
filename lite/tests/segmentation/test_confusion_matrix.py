from valor_lite.segmentation import DataLoader, MetricType, Segmentation


def test_confusion_matrix_basic_segmenations(
    basic_segmentations: list[Segmentation],
):
    loader = DataLoader()
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.ConfusionMatrix]]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "v1": {"v1": {"iou": 0.5}, "v2": {"iou": 0.0}},
                    "v2": {"v1": {"iou": 0.0}, "v2": {"iou": 0.5}},
                },
                "hallucinations": {
                    "v1": {"percent": 1.0},
                    "v2": {"percent": 0.0},
                },
                "missing_predictions": {
                    "v1": {"percent": 0.0},
                    "v2": {"percent": 0.0},
                },
            },
            "parameters": {},
        },
    ]
    for m in actual_metrics:
        import json

        print(json.dumps(m, indent=4))
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_segmentations_from_boxes(
    segmentations_from_boxes: list[Segmentation],
):
    loader = DataLoader()
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.ConfusionMatrix]]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": {"iou": 0.3333333333333333},  # 50% overlap
                        "v2": {"iou": 0.0},
                    },
                    "v2": {
                        "v1": {"iou": 0.0},
                        "v2": {
                            "iou": 5.000250012500625e-05  # overlaps 1 pixel out of 15,000 groundtruths
                        },
                    },
                },
                "hallucinations": {
                    "v1": {"percent": 0.0},
                    "v2": {"percent": 0.0},
                },
                "missing_predictions": {
                    "v1": {
                        "percent": 149.99  # overlaps 1 pixel out of 15,000 groundtruths
                    },
                    "v2": {"percent": 0.0},
                },
            },
            "parameters": {},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_large_random_segmentations(
    large_random_segmenations: list[Segmentation],
):
    loader = DataLoader()
    loader.add_data(large_random_segmenations)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(as_dict=True)[MetricType.ConfusionMatrix]
    assert len(metrics) == 1

    cm = metrics[0]["value"]["confusion_matrix"]
    hl = metrics[0]["value"]["hallucinations"]
    mp = metrics[0]["value"]["missing_predictions"]

    labels = set(cm.keys())

    for gt_label in labels:
        for pd_label in labels:
            if cm[gt_label][pd_label]["iou"] < 1e-9:
                cm[gt_label].pop(pd_label)
        if len(cm[gt_label]) == 0:
            cm.pop(gt_label)

    for label in labels:
        if hl[label]["percent"] < 1e-9:
            hl.pop(label)
        if mp[label]["percent"] < 1e-9:
            mp.pop(label)

    import json

    print(json.dumps(metrics[0], indent=4))
    assert cm == {}
