from pathlib import Path
from random import choice, uniform
from uuid import uuid4

import pyarrow.compute as pc

from valor_lite.object_detection import BoundingBox, Detection, Loader


def _generate_random_detections(
    n_detections: int, n_boxes: int, labels: str
) -> list[Detection]:
    def bbox(is_prediction):
        width, height = 50, 50
        xmin, ymin = uniform(0, 1000), uniform(0, 1000)
        xmax, ymax = uniform(xmin, xmin + width), uniform(ymin, ymin + height)
        kw = {"scores": [uniform(0, 1)]} if is_prediction else {}
        return BoundingBox(
            str(uuid4()),
            xmin,
            xmax,
            ymin,
            ymax,
            [choice(labels)],
            metadata=None,
            **kw,
        )

    return [
        Detection(
            uid=f"uid{i}",
            groundtruths=[bbox(is_prediction=False) for _ in range(n_boxes)],
            predictions=[bbox(is_prediction=True) for _ in range(n_boxes)],
        )
        for i in range(n_detections)
    ]


def test_fuzz_detections(loader: Loader):

    n_detections = 1
    n_boxes = 300
    labels = "abcdefghijklmnopqrstuvwxyz123456789"

    detections = _generate_random_detections(n_detections, n_boxes, labels)

    loader.add_bounding_boxes(detections)
    evaluator = loader.finalize()
    evaluator.compute_precision_recall(
        iou_thresholds=[0.25, 0.75],
        score_thresholds=[0.25, 0.75],
    )
    evaluator.compute_confusion_matrix_with_examples(
        iou_thresholds=[0.25, 0.75],
        score_thresholds=[0.25, 0.75],
    )
    # evaluator.delete()


def test_fuzz_detections_with_filtering(loader: Loader, tmp_path: Path):

    labels = "abcdefghijklmnopqrstuvwxyz123456789"
    n_detections = 10
    n_boxes = 10

    detections = _generate_random_detections(n_detections, n_boxes, labels)

    loader.add_bounding_boxes(detections)
    evaluator = loader.finalize()

    datum_subset = [f"uid{i}" for i in range(len(detections) // 2)]
    filtered_evaluator = evaluator.filter(
        datums=pc.field("datum_uid").isin(datum_subset),
        path=tmp_path / "filtered",
    )
    filtered_evaluator.compute_precision_recall(
        iou_thresholds=[0.25, 0.75],
        score_thresholds=[0.25, 0.75],
    )
    filtered_evaluator.compute_confusion_matrix_with_examples(
        iou_thresholds=[0.25, 0.75],
        score_thresholds=[0.25, 0.75],
    )
    filtered_evaluator.delete()
    evaluator.delete()


def test_fuzz_confusion_matrix(loader: Loader):
    dets = _generate_random_detections(10, 30, "abcde")
    loader.add_bounding_boxes(dets)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 10
    assert evaluator.info.number_of_labels == 5
    assert evaluator.info.number_of_groundtruth_annotations == 300
    assert evaluator.info.number_of_prediction_annotations == 300

    evaluator.compute_precision_recall(
        iou_thresholds=[0.25, 0.75],
        score_thresholds=[0.25, 0.75],
    )
    evaluator.compute_confusion_matrix_with_examples(
        iou_thresholds=[0.25, 0.75],
        score_thresholds=[0.25, 0.75],
    )
