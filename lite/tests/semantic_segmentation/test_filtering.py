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
        "number_of_groundtruths": 2,
        "number_of_predictions": 2,
        "number_of_groundtruth_pixels": 540000,
        "number_of_prediction_pixels": 540000,
    }

    assert evaluator.n_datums == 2
    assert (
        evaluator._label_metadata == np.array([[10000, 10000], [15000, 5000]])
    ).all()

    # test datum filtering
    filter_by_uid1 = evaluator.create_filter(datum_uids=["uid1"])
    assert filter_by_uid1.indices == np.array([0])
    assert (
        filter_by_uid1.label_metadata == np.array([[10000, 10000], [0, 0]])
    ).all()

    filter_by_uid2 = evaluator.create_filter(datum_uids=["uid2"])
    assert filter_by_uid2.indices == np.array([1])
    assert (
        filter_by_uid2.label_metadata == np.array([[0, 0], [15000, 5000]])
    ).all()

    # test label filtering
    filter_by_label_v1 = evaluator.create_filter(labels=["v1"])
    assert (filter_by_label_v1.indices == np.array([0, 1])).all()
    assert (
        filter_by_label_v1.label_metadata == np.array([[10000, 10000], [0, 0]])
    ).all()

    filter_by_label_v2 = evaluator.create_filter(labels=["v2"])
    assert (filter_by_label_v2.indices == np.array([0, 1])).all()
    assert (
        filter_by_label_v2.label_metadata == np.array([[0, 0], [15000, 5000]])
    ).all()

    # test joint filtering
    filter_by_uid0_v1 = evaluator.create_filter(
        datum_uids=["uid1"], labels=["v1"]
    )
    assert filter_by_uid0_v1.indices == np.array([0])
    assert (
        filter_by_uid0_v1.label_metadata == np.array([[10000, 10000], [0, 0]])
    ).all()

    filter_by_uid0_v2 = evaluator.create_filter(
        datum_uids=["uid1"], labels=["v2"]
    )
    assert filter_by_uid0_v2.indices == np.array([0])
    assert (
        filter_by_uid0_v2.label_metadata == np.array([[0, 0], [0, 0]])
    ).all()

    # test filter all
    filter_by_all = evaluator.create_filter(datum_uids=[])
    assert filter_by_all.indices.size == 0
    assert (
        filter_by_uid0_v2.label_metadata == np.array([[0, 0], [0, 0]])
    ).all()
