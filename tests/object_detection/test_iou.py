import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from valor_lite.object_detection.computation import (
    compute_bbox_iou,
    compute_bitmask_iou,
    compute_polygon_iou,
)


def test_compute_bbox_iou():

    # xmin, xmax, ymin, ymax
    box1 = np.array([0.0, 10.0, 0.0, 10.0])
    box2 = np.array([5.0, 10.0, 0.0, 10.0])
    box3 = np.array([0.0, 5.0, 5.0, 10.0])
    box4 = np.array([5.0, 15.0, 0.0, 10.0])
    box5 = np.array([0.0, 15.0, 0.0, 10.0])

    pairs = np.array(
        [
            np.stack((box1, box1), axis=0),
            np.stack((box1, box2), axis=0),
            np.stack((box1, box3), axis=0),
            np.stack((box1, box4), axis=0),
            np.stack((box1, box5), axis=0),
        ]
    )

    ious = compute_bbox_iou(pairs)
    assert len(ious) == 5
    assert ious[0] == 1.0
    assert ious[1] == 0.5
    assert ious[2] == 0.25
    assert round(ious[3], 5) == 0.33333
    assert round(ious[4], 5) == 0.66667

    # test no data
    assert (
        compute_bbox_iou(data=np.array([], dtype=np.float64))
        == np.array([], dtype=np.float64)
    ).all()


def test_compute_bitmask_iou():
    empty_10x10 = np.zeros((10, 10), dtype=np.bool_)
    filled_10x10 = np.ones((10, 10), dtype=np.bool_)

    top_left_corner = empty_10x10.copy()
    top_left_corner[:5, :5] = True

    bottom_right_corner = empty_10x10.copy()
    bottom_right_corner[5:, 5:] = True

    checker = top_left_corner | bottom_right_corner

    diagonal = np.eye(10, 10, dtype=np.bool_)

    gt_bitmasks = [
        filled_10x10,
        filled_10x10,
        top_left_corner,
        bottom_right_corner,
        checker,
        diagonal,
    ]

    pd_bitmasks = [
        filled_10x10,
        top_left_corner,
        bottom_right_corner,
        checker,
        filled_10x10,
        filled_10x10,
    ]

    data = np.array(list(zip(gt_bitmasks, pd_bitmasks)), dtype=np.bool_)

    result = compute_bitmask_iou(data=data)

    assert (result == [1.0, 0.25, 0.0, 0.5, 0.5, 0.1]).all()

    # test no data
    assert (
        compute_bitmask_iou(data=np.array([], dtype=np.bool_))
        == np.array([], dtype=np.float64)
    ).all()


def test_compute_polygon_iou():
    """Test ability to calculate IOU for rotated bounding boxes represented as polygons."""

    tests = [
        {
            "original_bbox1": [(1, 1), (6, 1), (6, 6), (1, 6)],
            "original_bbox2": [(3, 3), (8, 3), (8, 8), (3, 8)],
            "angles": [30, 60, 90, 112, 157, 249, 312],
            "bbox1": [
                [
                    (2.584936490538903, 0.08493649053890318),
                    (6.915063509461096, 2.5849364905389027),
                    (4.415063509461097, 6.915063509461096),
                    (0.08493649053890362, 4.415063509461096),
                    (2.584936490538903, 0.08493649053890318),
                ],
                [
                    (4.415063509461096, 0.08493649053890318),
                    (6.915063509461097, 4.415063509461096),
                    (2.5849364905389036, 6.915063509461096),
                    (0.08493649053890273, 2.5849364905389036),
                    (4.415063509461096, 0.08493649053890318),
                ],
                [(6.0, 1.0), (6.0, 6.0), (1.0, 6.0), (1.0, 1.0), (6.0, 1.0)],
                [
                    (6.754476119956748, 2.118556847122812),
                    (4.881443152877187, 6.754476119956749),
                    (0.2455238800432502, 4.881443152877188),
                    (2.118556847122811, 0.2455238800432511),
                    (6.754476119956748, 2.118556847122812),
                ],
                [
                    (6.778089954854287, 4.824434312407915),
                    (2.175565687592086, 6.778089954854286),
                    (0.221910045145715, 2.1755656875920844),
                    (4.8244343124079165, 0.22191004514571322),
                    (6.778089954854287, 4.824434312407915),
                ],
                [
                    (2.061968807620248, 6.729870940106256),
                    (0.27012905989374447, 2.0619688076202483),
                    (4.938031192379752, 0.27012905989374403),
                    (6.729870940106256, 4.938031192379752),
                    (2.061968807620248, 6.729870940106256),
                ],
                [
                    (-0.030688579590631093, 3.6850355477963417),
                    (3.3149644522036583, -0.030688579590631537),
                    (7.030688579590632, 3.3149644522036574),
                    (3.6850355477963417, 7.030688579590631),
                    (-0.030688579590631093, 3.6850355477963417),
                ],
            ],
            "bbox2": [
                [
                    (4.584936490538903, 2.084936490538903),
                    (8.915063509461095, 4.584936490538903),
                    (6.415063509461096, 8.915063509461095),
                    (2.0849364905389027, 6.4150635094610955),
                    (4.584936490538903, 2.084936490538903),
                ],
                [
                    (6.4150635094610955, 2.084936490538903),
                    (8.915063509461095, 6.4150635094610955),
                    (4.584936490538904, 8.915063509461097),
                    (2.0849364905389027, 4.5849364905389045),
                    (6.4150635094610955, 2.084936490538903),
                ],
                [(8.0, 3.0), (8.0, 8.0), (3.0, 8.0), (3.0, 3.0), (8.0, 3.0)],
                [
                    (8.754476119956747, 4.118556847122812),
                    (6.881443152877187, 8.754476119956749),
                    (2.245523880043251, 6.881443152877189),
                    (4.118556847122811, 2.2455238800432515),
                    (8.754476119956747, 4.118556847122812),
                ],
                [
                    (8.778089954854286, 6.824434312407915),
                    (4.175565687592085, 8.778089954854286),
                    (2.221910045145714, 4.175565687592085),
                    (6.824434312407915, 2.221910045145714),
                    (8.778089954854286, 6.824434312407915),
                ],
                [
                    (4.061968807620248, 8.729870940106256),
                    (2.270129059893745, 4.061968807620248),
                    (6.938031192379753, 2.270129059893746),
                    (8.729870940106256, 6.938031192379753),
                    (4.061968807620248, 8.729870940106256),
                ],
                [
                    (1.9693114204093698, 5.685035547796343),
                    (5.314964452203658, 1.9693114204093694),
                    (9.030688579590631, 5.314964452203658),
                    (5.685035547796342, 9.030688579590631),
                    (1.9693114204093698, 5.685035547796343),
                ],
            ],
            "expected": [
                0.2401,
                0.2401,
                0.2195,
                0.2295,
                0.2306,
                0.2285,
                0.2676,
            ],
        },
        {
            "original_bbox1": [(12, 15), (45, 15), (45, 48), (12, 48)],
            "original_bbox2": [(22, 25), (55, 25), (55, 58), (22, 58)],
            "angles": [
                7,
                24,
                40,
                65,
                84,
                107,
                120,
                143,
                167,
            ],
            "bbox1": [
                [
                    (14.13383266410312, 13.112144331733255),
                    (46.88785566826675, 17.13383266410312),
                    (42.866167335896876, 49.88785566826675),
                    (10.112144331733253, 45.866167335896876),
                    (14.13383266410312, 13.112144331733255),
                ],
                [
                    (20.13765455964779, 9.715345338146385),
                    (50.28465466185362, 23.13765455964779),
                    (36.862345440352215, 53.28465466185362),
                    (6.715345338146383, 39.862345440352215),
                    (20.13765455964779, 9.715345338146385),
                ],
                [
                    (26.466262248364764, 8.254271128708965),
                    (51.745728871291035, 29.46626224836476),
                    (30.53373775163524, 54.745728871291035),
                    (5.254271128708968, 33.53373775163524),
                    (26.466262248364764, 8.254271128708965),
                ],
                [
                    (36.480877167383184, 9.572720195173734),
                    (50.42727980482626, 39.480877167383184),
                    (20.519122832616816, 53.427279804826256),
                    (6.572720195173737, 23.519122832616812),
                    (36.480877167383184, 9.572720195173734),
                ],
                [
                    (43.18489162966023, 13.365669082507209),
                    (46.63433091749279, 46.18489162966023),
                    (13.815108370339779, 49.63433091749279),
                    (10.36566908250721, 16.81510837033977),
                    (43.18489162966023, 13.365669082507209),
                ],
                [
                    (49.10316160131525, 20.54510465453507),
                    (39.45489534546494, 52.10316160131524),
                    (7.896838398684764, 42.454895345464934),
                    (17.545104654535074, 10.89683839868476),
                    (49.10316160131525, 20.54510465453507),
                ],
                [
                    (51.03941916244324, 25.46058083755676),
                    (34.539419162443245, 54.03941916244324),
                    (5.960580837556762, 37.53941916244324),
                    (22.460580837556762, 8.960580837556762),
                    (51.03941916244324, 25.46058083755676),
                ],
                [
                    (51.60743379778914, 34.74753803377154),
                    (25.252461966228466, 54.60743379778913),
                    (5.3925662022108725, 28.252461966228463),
                    (31.747538033771548, 8.392566202210872),
                    (51.60743379778914, 34.74753803377154),
                ],
                [
                    (48.288798465630144, 43.865413672282614),
                    (16.13458632771738, 51.28879846563015),
                    (8.711201534369835, 19.134586327717386),
                    (40.86541367228261, 11.711201534369849),
                    (48.288798465630144, 43.865413672282614),
                ],
            ],
            "bbox2": [
                [
                    (24.133832664103117, 23.11214433173326),
                    (56.887855668266745, 27.133832664103124),
                    (52.866167335896876, 59.88785566826675),
                    (20.11214433173325, 55.86616733589688),
                    (24.133832664103117, 23.11214433173326),
                ],
                [
                    (30.137654559647785, 19.71534533814638),
                    (60.28465466185362, 33.137654559647785),
                    (46.862345440352215, 63.28465466185361),
                    (16.71534533814638, 49.862345440352215),
                    (30.137654559647785, 19.71534533814638),
                ],
                [
                    (36.46626224836476, 18.254271128708965),
                    (61.745728871291035, 39.46626224836476),
                    (40.53373775163524, 64.74572887129104),
                    (15.254271128708968, 43.53373775163524),
                    (36.46626224836476, 18.254271128708965),
                ],
                [
                    (46.480877167383184, 19.572720195173737),
                    (60.42727980482627, 49.480877167383184),
                    (30.51912283261682, 63.42727980482626),
                    (16.572720195173737, 33.519122832616816),
                    (46.480877167383184, 19.572720195173737),
                ],
                [
                    (53.18489162966023, 23.36566908250721),
                    (56.63433091749279, 56.18489162966023),
                    (23.81510837033978, 59.63433091749279),
                    (20.365669082507218, 26.81510837033977),
                    (53.18489162966023, 23.36566908250721),
                ],
                [
                    (59.10316160131525, 30.54510465453507),
                    (49.454895345464934, 62.10316160131524),
                    (17.896838398684764, 52.454895345464934),
                    (27.545104654535074, 20.89683839868476),
                    (59.10316160131525, 30.54510465453507),
                ],
                [
                    (61.03941916244324, 35.46058083755676),
                    (44.53941916244324, 64.03941916244324),
                    (15.960580837556762, 47.53941916244324),
                    (32.46058083755676, 18.960580837556765),
                    (61.03941916244324, 35.46058083755676),
                ],
                [
                    (61.60743379778913, 44.74753803377154),
                    (35.25246196622846, 64.60743379778913),
                    (15.392566202210872, 38.252461966228466),
                    (41.74753803377154, 18.392566202210872),
                    (61.60743379778913, 44.74753803377154),
                ],
                [
                    (58.28879846563015, 53.865413672282614),
                    (26.134586327717386, 61.28879846563015),
                    (18.71120153436985, 29.134586327717386),
                    (50.865413672282614, 21.71120153436984),
                    (58.28879846563015, 53.865413672282614),
                ],
            ],
            "expected": [
                0.3224,
                0.3403,
                0.3809,
                0.3421,
                0.3219,
                0.3303,
                0.3523,
                0.3711,
                0.3263,
            ],
        },
    ]

    for test in tests:
        gt_polygons = [
            ShapelyPolygon(coordinates) for coordinates in test["bbox1"]
        ]
        pd_polygons = [
            ShapelyPolygon(coordinates) for coordinates in test["bbox2"]
        ]

        data = np.array(list(zip(gt_polygons, pd_polygons)), dtype=object)
        result = compute_polygon_iou(data=data)
        assert (np.round(result, 4) == test["expected"]).all()

    # test no data
    assert (
        compute_polygon_iou(data=np.array([], dtype=object))
        == np.array([], dtype=np.float64)
    ).all()
