import numpy as np
from valor_lite.object_detection import (
    DataLoader,
    Detection,
    generate_bounding_box,
    generate_bounding_box_pair,
)
from valor_lite.profiling import Benchmark, create_runtime_profiler


def benchmark_add_bounding_boxes(
    n_labels: int,
    n_annotation_pairs: int,
    n_annotation_unmatched: int,
    time_limit: float | None,
    repeat: int = 1,
):

    profile = create_runtime_profiler(
        time_limit=time_limit,
        repeat=repeat,
    )

    elapsed = 0
    n_matched, n_unmatched = n_annotations_per_datum
    for _ in range(repeat):
        gts = []
        pds = []
        for _ in range(n_matched):
            gt, pd = generate_bounding_box_pair(n_labels)
            gts.append(gt)
            pds.append(pd)
        for _ in range(n_unmatched):
            gt = generate_bounding_box(n_labels, is_prediction=False)
            gts.append(gt)
            pd = generate_bounding_box(n_labels, is_prediction=True)
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
    n_annotations_per_datum: tuple[int, int],
    time_limit: float | None,
    repeat: int = 1,
):

    profile = create_runtime_profiler(
        time_limit=time_limit,
        repeat=repeat,
    )

    elapsed = 0
    for _ in range(repeat):

        n_matched, n_unmatched = n_annotations_per_datum
        pairs = [
            generate_bounding_box_pair(n_labels=n_labels)
            for _ in range(n_matched)
        ]
        unmatched_gts = [
            generate_bounding_box(n_labels=n_labels, is_prediction=False)
            for _ in range(n_unmatched)
        ]
        unmatched_pds = [
            generate_bounding_box(n_labels=n_labels, is_prediction=True)
            for _ in range(n_unmatched)
        ]
        gts = [gt for gt, _ in pairs] + unmatched_gts
        pds = [pd for _, pd in pairs] + unmatched_pds

        loader = DataLoader()
        for i in range(n_datums):
            detection = Detection(
                uid=str(i),
                groundtruths=gts,
                predictions=pds,
            )
            loader.add_bounding_boxes([detection])
        elapsed += profile(loader.finalize)()
    return elapsed / repeat


def benchmark_compute_precision_recall(
    n_datums: int,
    n_labels: int,
    n_annotations_per_datum: tuple[int, int],
    time_limit: float | None,
    repeat: int = 1,
):

    profile = create_runtime_profiler(
        time_limit=time_limit,
        repeat=repeat,
    )

    elapsed = 0
    for _ in range(repeat):

        n_matched, n_unmatched = n_annotations_per_datum
        pairs = [
            generate_bounding_box_pair(n_labels=n_labels)
            for _ in range(n_matched)
        ]
        unmatched_gts = [
            generate_bounding_box(n_labels=n_labels, is_prediction=False)
            for _ in range(n_unmatched)
        ]
        unmatched_pds = [
            generate_bounding_box(n_labels=n_labels, is_prediction=True)
            for _ in range(n_unmatched)
        ]
        gts = [gt for gt, _ in pairs] + unmatched_gts
        pds = [pd for _, pd in pairs] + unmatched_pds

        loader = DataLoader()
        for i in range(n_datums):
            detection = Detection(
                uid=str(i),
                groundtruths=gts,
                predictions=pds,
            )
            loader.add_bounding_boxes([detection])
        evaluator = loader.finalize()
        elapsed += profile(evaluator.compute_precision_recall)()
    return elapsed / repeat


def benchmark_compute_confusion_matrix(
    n_datums: int,
    n_labels: int,
    n_annotations_per_datum: tuple[int, int],
    n_examples: int,
    time_limit: float | None,
    repeat: int = 1,
):

    profile = create_runtime_profiler(
        time_limit=time_limit,
        repeat=repeat,
    )

    elapsed = 0
    for _ in range(repeat):

        n_matched, n_unmatched = n_annotations_per_datum
        pairs = [
            generate_bounding_box_pair(n_labels=n_labels)
            for _ in range(n_matched)
        ]
        unmatched_gts = [
            generate_bounding_box(n_labels=n_labels, is_prediction=False)
            for _ in range(n_unmatched)
        ]
        unmatched_pds = [
            generate_bounding_box(n_labels=n_labels, is_prediction=True)
            for _ in range(n_unmatched)
        ]
        gts = [gt for gt, _ in pairs] + unmatched_gts
        pds = [pd for _, pd in pairs] + unmatched_pds

        loader = DataLoader()
        for i in range(n_datums):
            detection = Detection(
                uid=str(i),
                groundtruths=gts,
                predictions=pds,
            )
            loader.add_bounding_boxes([detection])
        evaluator = loader.finalize()
        elapsed += profile(evaluator.compute_confusion_matrix)(
            number_of_examples=n_examples
        )
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

    n_annotations_per_datum = [
        (1000, 1),
        (100, 10),
        (10, 2),
    ]

    n_examples = [
        10,
        5,
        1,
        0,
    ]

    b = Benchmark(
        time_limit=5.0,
        memory_limit=4 * (1024**3),
        repeat=1,
        verbose=True,
    )

    b.run(
        benchmark=benchmark_add_bounding_boxes,
        n_labels=n_labels,
        n_annotations_per_datum=n_annotations_per_datum,
    )

    b.run(
        benchmark=benchmark_finalize,
        n_datums=n_datums,
        n_labels=n_labels,
        n_annotations_per_datum=n_annotations_per_datum,
    )

    b.run(
        benchmark=benchmark_compute_precision_recall,
        n_datums=n_datums,
        n_labels=n_labels,
        n_annotations_per_datum=n_annotations_per_datum,
    )
