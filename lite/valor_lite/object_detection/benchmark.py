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

    n_matched, n_unmatched = n_boxes_per_datum

    elapsed = 0
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

    n_matched, n_unmatched = n_boxes_per_datum

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

    elapsed = 0
    for _ in range(repeat):
        loader = DataLoader()
        for i in range(n_datums):
            detection = Detection(
                uid=f"uid{i}",
                groundtruths=gts,
                predictions=pds,
            )
            loader.add_bounding_boxes([detection])
        elapsed += profile(loader.finalize)()
    return elapsed / repeat


if __name__ == "__main__":

    n_datums = [
        100,
        10,
        1,
    ]

    n_labels = [
        # 1000,
        100,
        10,
        1,
    ]

    n_boxes_per_datum = [
        (100, 1),
        (10, 10),
        (1, 100),
    ]

    b = Benchmark(
        time_limit=10.0,
        memory_limit=8 * (1024**3),
        repeat=1,
        verbose=True,
    )

    # b.run(
    #     benchmark=benchmark_add_bounding_boxes,
    #     n_labels=n_labels,
    #     n_boxes_per_datum=n_boxes_per_datum,
    # )

    b.run(
        benchmark=benchmark_finalize,
        n_datums=n_datums,
        n_labels=n_labels,
        n_boxes_per_datum=n_boxes_per_datum,
    )

    # b.run(
    #     benchmark=benchmark_finalize,
    #     n_datums=n_datums,
    #     n_labels=n_labels,
    # )

    # b.run(
    #     benchmark=benchmark_evaluate,
    #     n_datums=n_datums,
    #     n_labels=n_labels,
    # )
