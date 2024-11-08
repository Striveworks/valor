import math
import random

import numpy as np
from valor_lite.object_detection import (
    Bitmask,
    BoundingBox,
    DataLoader,
    Detection,
    Polygon,
)
from valor_lite.profiling import Benchmark, create_runtime_profiler


def generate_random_bbox(
    n_labels: int,
    is_prediction: bool,
) -> BoundingBox:

    scale = random.uniform(25, 100)
    offset_x = random.uniform(0, 10000)
    offset_y = random.uniform(0, 10000)

    side_length = random.uniform(0.1, 0.5)

    xmax = max(1 - side_length, 0)
    ymax = max(1 - side_length, 0)
    x = random.uniform(0, xmax)
    y = random.uniform(0, ymax)

    xmin0 = x * scale + offset_x
    xmax0 = (x + side_length) * scale + offset_x
    ymin0 = y * scale + offset_y
    ymax0 = (y + side_length) * scale + offset_y

    if n_labels > 1:
        if not is_prediction:
            gt_label = str(random.randint(0, n_labels - 1))
            return BoundingBox(
                xmin=xmin0,
                xmax=xmax0,
                ymin=ymin0,
                ymax=ymax0,
                labels=[gt_label],
            )
        else:
            labels = [str(i) for i in range(n_labels)]
            common_proba = 0.4 / (n_labels - 1)
            scores = [0.5] + [common_proba for _ in range(n_labels - 1)]
            return BoundingBox(
                xmin=xmin0,
                xmax=xmax0,
                ymin=ymin0,
                ymax=ymax0,
                labels=labels,
                scores=scores,
            )
    elif n_labels == 1:
        if not is_prediction:
            return BoundingBox(
                xmin=xmin0,
                xmax=xmax0,
                ymin=ymin0,
                ymax=ymax0,
                labels=["0"],
            )
        else:
            pd_score = random.uniform(0.1, 0.9)
            return BoundingBox(
                xmin=xmin0,
                xmax=xmax0,
                ymin=ymin0,
                ymax=ymax0,
                labels=["0"],
                scores=[pd_score],
            )
    else:
        raise ValueError


def generate_random_bbox_pair(
    n_labels: int,
) -> tuple[BoundingBox, BoundingBox]:

    scale = random.uniform(25, 100)
    offset_x = random.uniform(0, 10000)
    offset_y = random.uniform(0, 10000)

    iou = random.uniform(0.1, 0.9)
    side_length = random.uniform(0.1, 0.5)
    intersection_area = (2 * iou * side_length * side_length) / (1 + iou)
    delta = side_length - math.sqrt(intersection_area)

    xmax = max(1 - side_length - delta, 0)
    ymax = max(1 - side_length - delta, 0)
    x = random.uniform(0, xmax)
    y = random.uniform(0, ymax)

    xmin0 = x * scale + offset_x
    xmax0 = (x + side_length) * scale + offset_x
    ymin0 = y * scale + offset_y
    ymax0 = (y + side_length) * scale + offset_y

    xmin1 = (x + delta) * scale + offset_x
    xmax1 = (x + delta + side_length) * scale + offset_x
    ymin1 = (y + delta) * scale + offset_y
    ymax1 = (y + delta + side_length) * scale + offset_y

    if n_labels > 1:
        common_proba = 0.4 / (n_labels - 1)
        labels = [str(i) for i in range(n_labels)]
        scores = [0.5] + [common_proba for _ in range(n_labels - 1)]
        gt_label = str(random.randint(0, n_labels - 1))
        gt = BoundingBox(
            xmin=xmin0,
            xmax=xmax0,
            ymin=ymin0,
            ymax=ymax0,
            labels=[gt_label],
        )
        pd = BoundingBox(
            xmin=xmin1,
            xmax=xmax1,
            ymin=ymin1,
            ymax=ymax1,
            labels=labels,
            scores=scores,
        )
    elif n_labels == 1:
        gt_label = str(random.randint(0, 1))
        pd_score = random.uniform(0.1, 0.9)
        gt = BoundingBox(
            xmin=xmin0,
            xmax=xmax0,
            ymin=ymin0,
            ymax=ymax0,
            labels=[gt_label],
        )
        pd = BoundingBox(
            xmin=xmin1,
            xmax=xmax1,
            ymin=ymin1,
            ymax=ymax1,
            labels=["0"],
            scores=[pd_score],
        )
    else:
        raise ValueError

    return (gt, pd)


def generate_cache(
    n_datums: int,
    n_labels: int,
    n_boxes_per_datum: tuple[int, int],
) -> DataLoader:
    """
    This skips the IOU computation.

    Not ideal since we are dealing directly with internals.
    """

    gts = []
    pds = []
    n_matched, n_unmatched = n_boxes_per_datum
    for _ in range(n_matched):
        gt, pd = generate_random_bbox_pair(n_labels)
        gts.append(gt)
        pds.append(pd)
    for _ in range(n_unmatched):
        gt = generate_random_bbox(n_labels, is_prediction=False)
        pd = generate_random_bbox(n_labels, is_prediction=True)
        gts.append(gt)
        pds.append(pd)

    detection = Detection(
        uid="0",
        groundtruths=gts,
        predictions=pds,
    )

    loader = DataLoader()
    loader.add_bounding_boxes([detection])

    # loader cache duplication
    assert len(loader.pairs) == 1

    # duplicate all iou pairs
    master_pair = loader.pairs[0]
    duplicated_pairs = list()
    for i in range(n_datums):
        duplicate_pair = master_pair.copy()
        duplicate_pair[:, 0] = i
        duplicated_pairs.append(duplicate_pair)
    loader.pairs = duplicated_pairs

    loader.groundtruth_count = {
        label_idx: {
            datum_idx: count * n_datums for datum_idx, count in values.items()
        }
        for label_idx, values in loader.groundtruth_count.items()
    }
    loader.prediction_count = {
        label_idx: {
            datum_idx: count * n_datums for datum_idx, count in values.items()
        }
        for label_idx, values in loader.prediction_count.items()
    }

    # evaluator cache duplication
    assert loader._evaluator.n_datums == 1
    loader._evaluator.n_datums = n_datums
    loader._evaluator.n_groundtruths = n_matched + n_unmatched
    loader._evaluator.n_predictions = n_matched + n_unmatched
    loader._evaluator.n_labels = n_labels
    loader._evaluator.uid_to_index = {str(i): i for i in range(n_datums)}
    loader._evaluator.index_to_uid = {i: str(i) for i in range(n_datums)}
    loader._evaluator.label_to_index = {str(i): i for i in range(n_labels)}
    loader._evaluator.index_to_label = {i: str(i) for i in range(n_labels)}

    return loader


def benchmark_add_bounding_boxes(
    n_labels: int,
    n_boxes_per_datum: tuple[int, int],
    time_limit: float | None,
    repeat: int = 1,
):

    profile = create_runtime_profiler(
        time_limit=time_limit,
        repeat=repeat,
    )

    elapsed = 0
    n_matched, n_unmatched = n_boxes_per_datum
    for _ in range(repeat):
        gts = []
        pds = []
        for _ in range(n_matched):
            gt, pd = generate_random_bbox_pair(n_labels)
            gts.append(gt)
            pds.append(pd)
        for _ in range(n_unmatched):
            gt = generate_random_bbox(n_labels, is_prediction=False)
            gts.append(gt)
            pd = generate_random_bbox(n_labels, is_prediction=True)
            pds.append(pd)

        detection = Detection(
            uid="uid",
            groundtruths=gts,
            predictions=pds,
        )
        loader = DataLoader()
        elapsed += profile(loader.add_bounding_boxes)([detection])
    return elapsed / repeat


def benchmark_finalize(
    n_datums: int,
    n_labels: int,
    n_boxes_per_datum: tuple[int, int],
    time_limit: float | None,
    repeat: int = 1,
):

    profile = create_runtime_profiler(
        time_limit=time_limit,
        repeat=repeat,
    )

    elapsed = 0
    for _ in range(repeat):
        loader = generate_cache(
            n_datums=n_datums,
            n_labels=n_labels,
            n_boxes_per_datum=n_boxes_per_datum,
        )
        elapsed += profile(loader.finalize)()
    return elapsed / repeat


if __name__ == "__main__":

    n_datums = [
        1000000,
        100000,
        10000,
        1000,
        100,
        10,
    ]

    n_labels = [
        1000,
        100,
        20,
        5,
    ]

    n_boxes_per_datum = [
        (1000, 1),
        (100, 10),
        (10, 2),
        (10, 100),
        (1, 1000),
    ]

    b = Benchmark(
        time_limit=10.0,
        memory_limit=8 * (1024**3),
        repeat=1,
        verbose=True,
    )

    b.run(
        benchmark=benchmark_add_bounding_boxes,
        n_labels=n_labels,
        n_boxes_per_datum=n_boxes_per_datum,
    )

    b.run(
        benchmark=benchmark_finalize,
        n_datums=n_datums,
        n_labels=n_labels,
        n_boxes_per_datum=n_boxes_per_datum,
    )

    # b.run(
    #     benchmark=benchmark_evaluate,
    #     n_datums=n_datums,
    #     n_labels=n_labels,
    # )

    loader = generate_cache(
        n_datums=1,
        n_labels=10,
        n_boxes_per_datum=(2, 0),
    )
    evaluator = loader.finalize()

    for pair in evaluator._detailed_pairs:
        print(pair.tolist())
