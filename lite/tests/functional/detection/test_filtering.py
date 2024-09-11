import numpy as np
from valor_lite.detection import DataLoader, Detection


def test_filtering(basic_detections: list[Detection]):
    """
    Basic object detection test.

    groundtruths
        datum uid1
            box 1 - label (k1, v1) - tp
            box 3 - label (k2, v2) - fn missing prediction
        datum uid2
            box 2 - label (k1, v1) - fn misclassification

    predictions
        datum uid1
            box 1 - label (k1, v1) - score 0.3 - tp
        datum uid2
            box 2 - label (k2, v2) - score 0.98 - fp
    """

    manager = DataLoader()
    manager.add_data(basic_detections)
    evaluator = manager.finalize()

    assert (
        evaluator._ranked_pairs
        == np.array(
            [
                [1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.98],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3],
            ]
        )
    ).all()

    # test datum filtering

    mask = evaluator.create_filter(datum_uids=["uid1"])
    assert (mask == np.array([False, True])).all()

    mask = evaluator.create_filter(datum_uids=["uid2"])
    assert (mask == np.array([True, False])).all()

    # test label filtering

    mask = evaluator.create_filter(labels=[("k1", "v1")])
    assert (mask == np.array([False, True])).all()

    mask = evaluator.create_filter(labels=[("k2", "v2")])
    assert (mask == np.array([False, False])).all()

    # test label key filtering

    mask = evaluator.create_filter(label_keys=["k1"])
    assert (mask == np.array([False, True])).all()

    mask = evaluator.create_filter(label_keys=["k2"])
    assert (mask == np.array([False, False])).all()

    # test combo
    mask = evaluator.create_filter(
        datum_uids=["uid1"],
        label_keys=["k1"],
    )
    assert (mask == np.array([False, True])).all()
