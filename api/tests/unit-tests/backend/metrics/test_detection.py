import pytest

from valor_api import schemas
from valor_api.backend.metrics.detection import (
    RankedPair,
    _calculate_101_pt_interp,
    _calculate_ap_and_ar,
    _compute_mean_detection_metrics_from_aps,
)


def truncate_float(x: float) -> str:
    return f"{int(x)}.{int((x - int(x)) * 100)}"


def test__calculate_101_pt_interp():
    # make sure we get back 0 if we don't pass any precisions
    assert _calculate_101_pt_interp([], []) == 0

    # get back -1 if all recalls and precisions are -1
    assert _calculate_101_pt_interp([-1, -1], [-1, -1]) == -1


def test__compute_mean_detection_metrics_from_aps():
    # make sure we get back 0 if we don't pass any precisions
    assert _compute_mean_detection_metrics_from_aps([]) == list()


def test__calculate_ap_and_ar():

    pairs = {
        "0": [
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="1",
                pd_datum_uid="1",
                gt_id=1,
                pd_id=1,
                score=0.8,
                iou=0.6,
                gt_geojson="",
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="1",
                pd_datum_uid="1",
                gt_id=2,
                pd_id=2,
                score=0.6,
                iou=0.8,
                gt_geojson="",
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="1",
                pd_datum_uid="1",
                gt_id=3,
                pd_id=3,
                score=0.4,
                iou=1.0,
                gt_geojson="",
                is_match=True,
            ),
        ],
        "1": [
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="1",
                pd_datum_uid="1",
                gt_id=0,
                pd_id=0,
                score=0.0,
                iou=1.0,
                gt_geojson="",
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="1",
                pd_datum_uid="1",
                gt_id=2,
                pd_id=2,
                score=0.0,
                iou=1.0,
                gt_geojson="",
                is_match=True,
            ),
        ],
        "2": [
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="1",
                pd_datum_uid="1",
                gt_id=0,
                pd_id=0,
                score=1.0,
                iou=1.0,
                gt_geojson="",
                is_match=True,
            ),
        ],
    }

    grouper_mappings = {
        "grouper_id_to_grouper_label_mapping": {
            "0": schemas.Label(key="name", value="car"),
            "1": schemas.Label(key="name", value="dog"),
            "2": schemas.Label(key="name", value="person"),
        }
    }

    number_of_groundtruths_per_grouper = {
        "0": 3,
        "1": 2,
        "2": 4,
    }

    iou_thresholds = [0.5, 0.75, 0.9]

    # Calculated by hand
    reference_ap_metrics = [
        schemas.APMetric(
            iou=0.5,
            value=1.0,
            label=schemas.Label(key="name", value="car", score=None),
        ),
        schemas.APMetric(
            iou=0.75,
            value=0.442244224422442,
            label=schemas.Label(key="name", value="car", score=None),
        ),
        schemas.APMetric(
            iou=0.9,
            value=0.11221122112211224,
            label=schemas.Label(key="name", value="car", score=None),
        ),
        schemas.APMetric(
            iou=0.5,
            value=0.0,
            label=schemas.Label(key="name", value="dog", score=None),
        ),
        schemas.APMetric(
            iou=0.75,
            value=0.0,
            label=schemas.Label(key="name", value="dog", score=None),
        ),
        schemas.APMetric(
            iou=0.9,
            value=0.0,
            label=schemas.Label(key="name", value="dog", score=None),
        ),
        schemas.APMetric(
            iou=0.5,
            value=0.25742574257425743,
            label=schemas.Label(key="name", value="person", score=None),
        ),
        schemas.APMetric(
            iou=0.75,
            value=0.25742574257425743,
            label=schemas.Label(key="name", value="person", score=None),
        ),
        schemas.APMetric(
            iou=0.9,
            value=0.25742574257425743,
            label=schemas.Label(key="name", value="person", score=None),
        ),
    ]

    reference_ar_metrics = [
        schemas.ARMetric(
            ious=set([0.5, 0.75, 0.9]),
            value=0.6666666666666666,  # average of [{'iou_threshold':.5, 'recall': 1}, {'iou_threshold':.75, 'recall':.66}, {'iou_threshold':.9, 'recall':.33}]
            label=schemas.Label(key="name", value="car", score=None),
        ),
        schemas.ARMetric(
            ious=set([0.5, 0.75, 0.9]),
            value=0.0,
            label=schemas.Label(key="name", value="dog", score=None),
        ),
        schemas.ARMetric(
            ious=set([0.5, 0.75, 0.9]),
            value=0.25,
            label=schemas.Label(key="name", value="person", score=None),
        ),
    ]

    ap_metrics, ar_metrics = _calculate_ap_and_ar(
        sorted_ranked_pairs=pairs,
        number_of_groundtruths_per_grouper=number_of_groundtruths_per_grouper,
        grouper_mappings=grouper_mappings,
        iou_thresholds=iou_thresholds,
        recall_score_threshold=0.0,
    )

    assert len(ap_metrics) == len(reference_ap_metrics)
    assert len(ar_metrics) == len(reference_ar_metrics)
    for pd, gt in zip(ap_metrics, reference_ap_metrics):
        assert pd.iou == gt.iou
        assert truncate_float(pd.value) == truncate_float(gt.value)
        assert pd.label == gt.label
    for pd, gt in zip(ar_metrics, reference_ar_metrics):
        assert pd.ious == gt.ious
        assert truncate_float(pd.value) == truncate_float(gt.value)
        assert pd.label == gt.label

    # Test iou threshold outside 0 < t <= 1
    for illegal_thresh in [-1.1, -0.1, 0, 1.1]:
        with pytest.raises(ValueError):
            _calculate_ap_and_ar(
                sorted_ranked_pairs=pairs,
                number_of_groundtruths_per_grouper=number_of_groundtruths_per_grouper,
                grouper_mappings=grouper_mappings,
                iou_thresholds=iou_thresholds + [0],
                recall_score_threshold=0.0,
            )

    # Test score threshold outside 0 <= t <= 1
    for illegal_thresh in [-1.1, -0.1, 1.1]:
        with pytest.raises(ValueError):
            _calculate_ap_and_ar(
                sorted_ranked_pairs=pairs,
                number_of_groundtruths_per_grouper=number_of_groundtruths_per_grouper,
                grouper_mappings=grouper_mappings,
                iou_thresholds=iou_thresholds,
                recall_score_threshold=illegal_thresh,
            )
