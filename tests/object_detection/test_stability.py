from random import choice, uniform
from uuid import uuid4

from valor_lite.object_detection import BoundingBox, DataLoader, Detection


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


def test_fuzz_detections():

    few_labels = "abc"
    many_labels = "abcdefghijklmnopqrstuvwxyz123456789"
    quantities = [1, 5, 10]

    for _ in range(100):

        labels = choice([few_labels, many_labels])
        n_detections = choice(quantities)
        n_boxes = choice(quantities)

        detections = _generate_random_detections(n_detections, n_boxes, labels)

        loader = DataLoader()
        loader.add_bounding_boxes(detections)
        evaluator = loader.finalize()
        evaluator.evaluate(
            iou_thresholds=[0.25, 0.75],
            score_thresholds=[0.25, 0.75],
        )


def test_fuzz_detections_with_filtering():

    few_labels = "abcd"
    many_labels = "abcdefghijklmnopqrstuvwxyz123456789"
    quantities = [4, 10]

    for _ in range(100):

        labels = choice([few_labels, many_labels])
        n_detections = choice(quantities)
        n_boxes = choice(quantities)

        detections = _generate_random_detections(n_detections, n_boxes, labels)

        loader = DataLoader()
        loader.add_bounding_boxes(detections)
        evaluator = loader.finalize()

        datum_subset = [f"uid{i}" for i in range(len(detections) // 2)]

        filter_ = evaluator.create_filter(datums=datum_subset)
        evaluator.evaluate(
            iou_thresholds=[0.25, 0.75],
            score_thresholds=[0.25, 0.75],
            filter_=filter_,
        )


def test_fuzz_confusion_matrix():
    dets = _generate_random_detections(1000, 30, "abcde")
    loader = DataLoader()
    loader.add_bounding_boxes(dets)
    evaluator = loader.finalize()
    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.metadata.to_dict() == {
        "number_of_datums": 1000,
        "number_of_ground_truths": 30000,
        "number_of_predictions": 30000,
        "number_of_labels": 5,
    }
    evaluator.evaluate(
        iou_thresholds=[0.25, 0.75],
        score_thresholds=[0.5],
    )
