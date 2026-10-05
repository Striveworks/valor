from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.compute as pc

from valor_lite.semantic_segmentation import Loader, Segmentation


def _generate_random_segmentations(
    n_segmentations: int, size_: int, n_labels: int
) -> list[Segmentation]:
    rng = np.random.default_rng(42)
    return [
        Segmentation(
            uid=f"uid{i}",
            groundtruths=rng.integers(
                n_labels, size=(size_, size_), dtype=np.uint16
            ),
            predictions=rng.integers(
                n_labels, size=(size_, size_), dtype=np.uint16
            ),
            labels=[str(value) for value in range(1, n_labels)],
        )
        for i in range(n_segmentations)
    ]


def test_fuzz_segmentations_with_filtering(loader: Loader, tmp_path: Path):
    segmentations = _generate_random_segmentations(10, 30, 5)
    expected = Counter()
    for segmentation in segmentations[:5]:
        expected.update(
            zip(
                segmentation.groundtruths.ravel().tolist(),
                segmentation.predictions.ravel().tolist(),
            )
        )
    assert len(expected) == 25
    loader.add_data(segmentations)
    evaluator = loader.finalize()
    subset = evaluator.filter(
        datums=pc.field("datum_uid").isin([f"uid{i}" for i in range(5)]),
        path=tmp_path / "subset",
    )
    matrix = subset._compute_confusion_matrix_intermediate()
    for (gt, pd), count in expected.items():
        assert matrix[gt, pd] == count
    assert matrix.sum() == 5 * 30 * 30
    assert subset.info.number_of_datums == 5
    subset.compute_precision_recall_iou()
