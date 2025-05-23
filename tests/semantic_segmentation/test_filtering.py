import numpy as np

from valor_lite.semantic_segmentation import DataLoader, Segmentation


def test_filtering(segmentations_from_boxes: list[Segmentation]):

    loader = DataLoader()
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    assert evaluator.metadata == {
        "ignored_prediction_labels": [],
        "missing_prediction_labels": [],
        "number_of_datums": 2,
        "number_of_labels": 2,
        "number_of_groundtruths": 25000,
        "number_of_predictions": 15000,
        "number_of_pixels": 540000,
        "is_filtered": False,
    }

    assert evaluator.n_datums == 2
    assert (
        evaluator._label_metadata == np.array([[10000, 10000], [15000, 5000]])
    ).all()

    # test datum filtering
    evaluator.apply_filter(datum_ids=["uid1"])
    assert evaluator.confusion_matrices.shape == (0, 0)
    assert (
        evaluator.label_metadata == np.array([[10000, 10000], [0, 0]])
    ).all()

    evaluator.apply_filter(datum_ids=["uid2"])
    assert filter_by_uid2.indices == np.array([1])
    assert (
        evaluator.label_metadata == np.array([[0, 0], [15000, 5000]])
    ).all()

    # test label filtering
    evaluator.apply_filter(labels=["v1"])
    assert (filter_by_label_v1.indices == np.array([0, 1])).all()
    assert (
        evaluator.label_metadata == np.array([[10000, 10000], [0, 0]])
    ).all()

    evaluator.apply_filter(labels=["v2"])
    assert (filter_by_label_v2.indices == np.array([0, 1])).all()
    assert (
        evaluator.label_metadata == np.array([[0, 0], [15000, 5000]])
    ).all()

    # test joint filtering
    evaluator.apply_filter(
        datum_ids=["uid1"], labels=["v1"]
    )
    assert filter_by_uid0_v1.indices == np.array([0])
    assert (
        evaluator.label_metadata == np.array([[10000, 10000], [0, 0]])
    ).all()

    evaluator.apply_filter(
        datum_ids=["uid1"], labels=["v2"]
    )
    assert filter_by_uid0_v2.indices == np.array([0])
    assert (
        evaluator.label_metadata == np.array([[0, 0], [0, 0]])
    ).all()

    # test filter all
    evaluator.apply_filter(datum_ids=[])
    assert filter_by_all.indices.size == 0
    assert (
        evaluator.label_metadata == np.array([[0, 0], [0, 0]])
    ).all()
