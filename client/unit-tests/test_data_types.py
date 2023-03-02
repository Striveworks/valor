from velour.data_types import rle_to_mask


def test_rle_to_mask():
    h, w = 4, 6
    rle = [(10, 4), (3, 7)]

    mask = rle_to_mask(run_length_encoding=rle, image_height=h, image_width=w)
    expected_mask = [
        [False, False, False, True, False, True],
        [False, False, False, True, True, True],
        [False, False, True, False, True, True],
        [False, False, True, False, True, True],
    ]

    assert mask.sum() == 4 + 7
    assert mask.tolist() == expected_mask
