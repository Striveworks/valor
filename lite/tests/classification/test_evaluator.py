from valor_lite.classification import Classification, DataLoader


def test_metadata_using_classification_example(
    classifications_animal_example: list[Classification],
):
    manager = DataLoader()
    manager.add_data(classifications_animal_example)
    evaluator = manager.finalize()

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 6
    assert evaluator.n_labels == 3
    assert evaluator.n_groundtruths == 6
    assert evaluator.n_predictions == 3 * 6

    assert evaluator.metadata == {
        "n_datums": 6,
        "n_groundtruths": 6,
        "n_predictions": 3 * 6,
        "n_labels": 3,
        "ignored_prediction_labels": [],
        "missing_prediction_labels": [],
    }
