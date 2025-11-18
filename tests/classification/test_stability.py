from pathlib import Path
from random import choice, uniform

import pyarrow.compute as pc
import pytest

from valor_lite.classification import Classification, Loader


def generate_random_classifications(
    n_classifications: int, n_labels: int
) -> list[Classification]:

    labels = [str(value) for value in range(n_labels)]
    scores = []
    running_sum = 0.0
    for _ in range(n_labels - 1):
        score = uniform(0, 1 - running_sum)
        running_sum += score
        scores.append(score)
    scores.append(1.0 - running_sum)

    return [
        Classification(
            uid=f"uid{i}",
            groundtruth=choice(labels),
            predictions=labels,
            scores=scores,
        )
        for i in range(n_classifications)
    ]


def test_fuzz_classifications(loader: Loader):
    if loader._writer.batch_size < 100:
        pytest.skip("batch size is too small to test filtering")

    n_classifications = 1000
    n_labels = 100

    classifications = generate_random_classifications(
        n_classifications, n_labels=n_labels
    )
    score_thresholds = [0.25, 0.75]

    loader.add_data(classifications)
    evaluator = loader.finalize()
    evaluator.compute_rocauc()
    evaluator.compute_precision_recall(
        score_thresholds=score_thresholds,
    )
    evaluator.compute_confusion_matrix(score_thresholds)


def test_fuzz_classifications_with_filtering(loader: Loader, tmp_path: Path):
    if loader._writer.batch_size < 100:
        pytest.skip("batch size is too small to test filtering")

    n_classifications = 1000
    n_labels = 100

    classifications = generate_random_classifications(
        n_classifications, n_labels=n_labels
    )

    loader.add_data(classifications)
    evaluator = loader.finalize()

    datum_subset = [f"uid{i}" for i in range(len(classifications) // 2)]

    filtered_evaluator = evaluator.filter(
        datums=pc.field("datum_uid").isin(datum_subset),
        path=tmp_path / "filter",
    )
    filtered_evaluator.compute_rocauc()
    filtered_evaluator.compute_precision_recall(
        score_thresholds=[0.25, 0.75],
    )
