from pathlib import Path
from random import choice

import numpy as np

from valor_lite.exceptions import EmptyFilterError
from valor_lite.semantic_segmentation import Bitmask, DataLoader, Segmentation


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


def test_fuzz_segmentations(tmp_path: Path):

    quantities = [1, 5, 10]
    sizes = [10, 100]

    for i in range(100):

        n_segmentations = choice(quantities)
        size_ = choice(sizes)
        n_labels = choice(quantities)

        segmentations = _generate_random_segmentations(
            n_segmentations, size_=size_, n_labels=n_labels
        )

        loader = DataLoader.create(tmp_path / str(i))
        loader.add_data(segmentations)
        evaluator = loader.finalize()
        evaluator.evaluate()


def test_fuzz_segmentations_with_filtering(tmp_path: Path):

    quantities = [4, 10]
    for i in range(100):

        n_segmentations = choice(quantities)
        size_ = choice(quantities)
        n_labels = choice(quantities)

        segmentations = _generate_random_segmentations(
            n_segmentations, size_=size_, n_labels=n_labels
        )

        loader = DataLoader.create(tmp_path / f"original_{i}")
        loader.add_data(segmentations)
        evaluator = loader.finalize()

        datum_subset = [f"uid{i}" for i in range(len(segmentations) // 2)]

        try:
            filter_ = evaluator.create_filter(datums=datum_subset)
        except EmptyFilterError:
            pass
        else:
            filtered_evaluator = evaluator.filter(
                tmp_path / f"filtered_{i}", filter_
            )
            filtered_evaluator.evaluate()
