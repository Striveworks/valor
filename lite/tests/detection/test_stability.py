from random import choice, uniform

from valor_lite.detection import BoundingBox, DataLoader, Detection


def generate_random_detections(
    n_detections: int, n_boxes: int, labels: str
) -> list[Detection]:
    def bbox(is_prediction):
        xmin, ymin = uniform(0, 10), uniform(0, 10)
        xmax, ymax = uniform(xmin, 15), uniform(ymin, 15)
        kw = (
            {"scores": [uniform(0, 1), uniform(0, 1)]} if is_prediction else {}
        )
        return BoundingBox(
            xmin,
            xmax,
            ymin,
            ymax,
            [("class", choice(labels)), ("category", choice(labels))],
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

        detections = generate_random_detections(n_detections, n_boxes, labels)

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

        detections = generate_random_detections(n_detections, n_boxes, labels)

        loader = DataLoader()
        loader.add_bounding_boxes(detections)
        evaluator = loader.finalize()

        label_key = "class"
        datum_subset = [f"uid{i}" for i in range(len(detections) // 2)]

        filter_ = evaluator.create_filter(
            datum_uids=datum_subset,
            label_keys=[label_key],
        )

        evaluator.evaluate(
            iou_thresholds=[0.25, 0.75],
            score_thresholds=[0.25, 0.75],
            filter_=filter_,
        )
