from valor_lite.segmentation import DataLoader, Segmentation


def test_metadata_using_large_random_segmentations(
    large_random_segmentations: list[Segmentation],
):
    manager = DataLoader()
    manager.add_data(large_random_segmentations)
    evaluator = manager.finalize()

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 3
    assert evaluator.n_labels == 9
    assert evaluator.n_groundtruths == 9
    assert evaluator.n_predictions == 9
    assert evaluator.n_groundtruth_pixels == 9000000
    assert evaluator.n_prediction_pixels == 9000000

    assert evaluator.metadata == {
        "ignored_prediction_labels": [],
        "missing_prediction_labels": [],
        "number_of_datums": 3,
        "number_of_labels": 9,
        "number_of_groundtruths": 9,
        "number_of_predictions": 9,
        "number_of_groundtruth_pixels": 9000000,
        "number_of_prediction_pixels": 9000000,
    }
