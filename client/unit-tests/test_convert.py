import PIL.Image

from velour.convert import chariot_detections_to_velour, coco_rle_to_mask
from velour.data_types import Image


def test_chariot_detections_to_velour():
    dets = {
        "num_detections": 2,
        "detection_classes": [
            "person",
            "person",
            "car",
        ],
        "detection_boxes": [
            [
                151.2235107421875,
                118.97279357910156,
                377.8422546386719,
                197.98605346679688,
            ],
            [
                94.09261322021484,
                266.5445556640625,
                419.3203430175781,
                352.9458923339844,
            ],
        ],
        "detection_scores": ["0.99932003", "0.99895525"],
    }

    velour_dets = chariot_detections_to_velour(
        dets, Image(uid="", width=10, height=100)
    )

    assert len(velour_dets) == 2
    assert [
        scored_label.label.key
        for det in velour_dets
        for scored_label in det.scored_labels
    ] == ["class", "class"]


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
