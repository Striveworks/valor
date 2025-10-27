from pathlib import Path
from random import choice, uniform

import pyarrow.compute as pc

from valor_lite.classification import Classification, DataLoader, Filter
from valor_lite.classification.loader import Loader


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


def test_fuzz_classifications(tmp_path: Path):

    for _ in range(10):

        n_classifications = 5000
        n_labels = 10

        classifications = generate_random_classifications(
            n_classifications, n_labels=n_labels
        )
        score_thresholds = [0.25, 0.75]

        print(classifications[0])

        loader = Loader.create(
            tmp_path,
            batch_size=500,
            rows_per_file=20_000,
            delete_if_exists=True,
        )
        loader.add_data(classifications)
        evaluator = loader.finalize()
        evaluator.compute_precision_recall_rocauc(
            score_thresholds=score_thresholds,
            read_batch_size=100,
        )
        evaluator.compute_confusion_matrix(score_thresholds)


def test_fuzz_classifications_with_filtering(tmp_path: Path):

    quantities = [4, 10]
    for _ in range(100):

        n_classifications = choice(quantities)
        n_labels = choice(quantities)

        classifications = generate_random_classifications(
            n_classifications, n_labels=n_labels
        )

        loader = DataLoader.create(tmp_path, delete_if_exists=True)
        loader.add_data(classifications)
        evaluator = loader.finalize()

        datum_subset = [f"uid{i}" for i in range(len(classifications) // 2)]

        filter_ = Filter(datums=pc.field("datum_uid").isin(datum_subset))
        evaluator.evaluate(
            score_thresholds=[0.25, 0.75],
            filter_=filter_,
        )
