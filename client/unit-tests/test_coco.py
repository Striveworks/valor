import PIL.Image

from velour.integrations.convert import coco_rle_to_mask


def test_coco_rle_to_mask():
    h, w = 4, 6
    coco_rle_seg_dict = {"counts": [10, 4, 3, 7], "size": (h, w)}

    mask = coco_rle_to_mask(coco_rle_seg_dict=coco_rle_seg_dict)
    expected_mask = [
        [False, False, False, True, False, True],
        [False, False, False, True, True, True],
        [False, False, True, False, True, True],
        [False, False, True, False, True, True],
    ]

    assert mask.sum() == 4 + 7
    assert mask.tolist() == expected_mask

    img = PIL.Image.fromarray(mask)

    assert img.width == w
    assert img.height == h
