from pathlib import Path

import numpy as np
import pyarrow.compute as pc

from valor_lite.semantic_segmentation import Bitmask, Loader, Segmentation


def _generate_random_segmentations(
    n_segmentations: int, size_: int, n_labels: int
) -> list[Segmentation]:

    labels = [str(value) for value in range(n_labels)]

    return [
        Segmentation(
            uid=f"uid{i}",
            groundtruths=[
                Bitmask(
                    mask=np.random.randint(
                        0, 1, size=(size_, size_), dtype=np.bool_
                    ),
                    label=label,
                )
                for label in labels
            ],
            predictions=[
                Bitmask(
                    mask=np.random.randint(
                        0, 1, size=(size_, size_), dtype=np.bool_
                    ),
                    label=label,
                )
                for label in labels
            ],
            shape=(size_, size_),
        )
        for i in range(n_segmentations)
    ]


def test_fuzz_segmentations(loader: Loader):

    n_segmentations = 10
    size_ = 100
    n_labels = 10

    segmentations = _generate_random_segmentations(
        n_segmentations, size_=size_, n_labels=n_labels
    )
    loader.add_data(segmentations)
    evaluator = loader.finalize()
    evaluator.compute_precision_recall_iou()


def test_fuzz_segmentations_with_filtering(loader: Loader, tmp_path: Path):

    n_segmentations = 15
    size_ = 100
    n_labels = 15

    segmentations = _generate_random_segmentations(
        n_segmentations, size_=size_, n_labels=n_labels
    )
    loader.add_data(segmentations)
    evaluator = loader.finalize()

    datum_subset = [f"uid{i}" for i in range(len(segmentations) // 2)]

    filtered_evaluator = evaluator.filter(
        datums=pc.field("datum_uid").isin(datum_subset),
        path=tmp_path / "filtered",
    )
    filtered_evaluator.compute_precision_recall_iou()
