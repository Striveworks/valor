from valor_lite.classification import Classification, DataLoader


def test_metadata_using_classification_example(
    classifications_two_categeories: list[Classification],
):
    manager = DataLoader()
    manager.add_data(classifications_two_categeories)
    evaluator = manager.finalize()

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 6
    assert evaluator.n_labels == 7
    assert evaluator.n_groundtruths == 12
    assert evaluator.n_predictions == 7 * 6

    assert evaluator.metadata == {
        "ignored_prediction_labels": [],
        "missing_prediction_labels": [],
        "n_datums": 6,
        "n_labels": 7,
        "n_groundtruths": 12,
        "n_predictions": 7 * 6,
    }
