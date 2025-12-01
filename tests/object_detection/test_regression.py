import json
from pathlib import Path

from valor_lite.object_detection import BoundingBox, Detection, Loader


def test_regression_coco_v0_36_6(loader: Loader):

    # load detections
    path = Path(__file__).parent / "fixtures/coco_input.json"
    with open(path, "r") as f:
        detections = json.load(f)
        detections = [
            Detection(
                uid=d["uid"],
                groundtruths=[BoundingBox(**gt) for gt in d["groundtruths"]],
                predictions=[BoundingBox(**pd) for pd in d["predictions"]],
            )
            for d in detections
        ]

    # load expected metrics
    path = Path(__file__).parent / "fixtures/coco_output.json"
    with open(path, "r") as f:
        expected_metrics = json.load(f)

    loader.add_bounding_boxes(detections)
    evaluator = loader.finalize()
    metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.1],
        score_thresholds=[0.5],
    )

    actual_metrics = {
        k.value: [m.to_dict() for m in v] for k, v in metrics.items()
    }

    # verify matching keys
    assert set(expected_metrics.keys()) == set(actual_metrics.keys())

    for mtype in expected_metrics.keys():
        for m in actual_metrics[mtype]:
            assert m in expected_metrics[mtype], mtype
        for m in expected_metrics[mtype]:
            assert m in actual_metrics[mtype], mtype
