from valor_lite.object_detection import BoundingBox, Detection, Loader


def test_regression_coco_v0_36_6(
    loader: Loader,
    coco_detections_v0_36_6: list[Detection[BoundingBox]],
    coco_metrics_v0_36_6: dict[str, list[dict]],
):
    """Test against Valor v0.36.6"""
    loader.add_bounding_boxes(coco_detections_v0_36_6)
    evaluator = loader.finalize()
    metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.1],
        score_thresholds=[0.5],
    )
    computed_metrics = {
        k.value: [m.to_dict() for m in v] for k, v in metrics.items()
    }

    # verify matching keys
    assert set(coco_metrics_v0_36_6.keys()) == set(computed_metrics.keys())

    for mtype in coco_metrics_v0_36_6.keys():
        for m in computed_metrics[mtype]:
            assert m in coco_metrics_v0_36_6[mtype], mtype
        for m in coco_metrics_v0_36_6[mtype]:
            assert m in computed_metrics[mtype], mtype
