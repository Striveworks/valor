from velour_api.metrics import _match_array


def test__match_array():
    ious = [
        [0.0, 0.004, 0.78],
        [0.65, 0.0, 0.0],
        [0.0, 0.8, 0.051],
        [0.71, 0.0, 0.0],
    ]

    assert _match_array(ious, 1.0) == [None, None, None, None]

    assert _match_array(ious, 0.75) == [2, None, 1, None]

    assert _match_array(ious, 0.7) == [2, None, 1, 0]

    # check that match to groundtruth 0 switches
    assert _match_array(ious, 0.1) == [2, 0, 1, None]

    assert _match_array(ious, 0.0) == [2, 0, 1, None]
