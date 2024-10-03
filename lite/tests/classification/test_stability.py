from random import choice, uniform

from valor_lite.classification import Classification, DataLoader


def generate_random_classifications(
    n_classifications: int, n_categories: int, n_labels: int
) -> list[Classification]:

    labels = [
        [(str(key), str(value)) for value in range(n_labels)]
        for key in range(n_categories)
    ]

    return [
        Classification(
            uid=f"uid{i}",
            groundtruths=[choice(category) for category in labels],
            predictions=[label for category in labels for label in category],
            scores=[uniform(0, 1) for _ in range(n_labels * n_categories)],
        )
        for i in range(n_classifications)
    ]


def test_fuzz_classifications():

    quantities = [1, 5, 10]
    for _ in range(100):

        n_classifications = choice(quantities)
        n_categories = choice(quantities)
        n_labels = choice(quantities)

        classifications = generate_random_classifications(
            n_classifications, n_categories=n_categories, n_labels=n_labels
        )

        loader = DataLoader()
        loader.add_data(classifications)
        evaluator = loader.finalize()
        evaluator.evaluate(
            score_thresholds=[0.25, 0.75],
        )


def test_fuzz_classifications_with_filtering():

    quantities = [4, 10]
    for _ in range(100):

        n_classifications = choice(quantities)
        n_categories = choice(quantities)
        n_labels = choice(quantities)

        classifications = generate_random_classifications(
            n_classifications, n_categories=n_categories, n_labels=n_labels
        )

        loader = DataLoader()
        loader.add_data(classifications)
        evaluator = loader.finalize()

        label_key = "1"
        datum_subset = [f"uid{i}" for i in range(len(classifications) // 2)]

        filter_ = evaluator.create_filter(
            datum_uids=datum_subset,
            label_keys=[label_key],
        )

        evaluator.evaluate(
            score_thresholds=[0.25, 0.75],
            filter_=filter_,
        )
