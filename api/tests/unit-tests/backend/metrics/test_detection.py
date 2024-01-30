from velour_api import schemas
from velour_api.backend.metrics.detection import (
    RankedPair,
    _ap,
    _calculate_101_pt_interp,
    _compute_mean_detection_metrics_from_aps,
)


def truncate_float(x: float) -> str:
    return f"{int(x)}.{int((x - int(x)) * 100)}"


def test__calculate_101_pt_interp():
    # make sure we get back 0 if we don't pass any precisions
    assert _calculate_101_pt_interp([], []) == 0


def test__compute_mean_detection_metrics_from_aps():
    # make sure we get back 0 if we don't pass any precisions
    assert _compute_mean_detection_metrics_from_aps([]) == list()


def test__ap():
    pairs = {
        "0": [
            RankedPair(1, 1, score=0.8, iou=0.6),
            RankedPair(2, 2, score=0.6, iou=0.8),
            RankedPair(3, 3, score=0.4, iou=1.0),
        ],
        "1": [
            RankedPair(0, 0, score=0.0, iou=1.0),
            RankedPair(2, 2, score=0.0, iou=1.0),
        ],
        "2": [
            RankedPair(0, 0, score=1.0, iou=1.0),
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
    reference_metrics = [
        schemas.APMetric(
            iou=0.5, value=1.0, label=schemas.Label(key="name", value="car")
        ),
        schemas.APMetric(
            iou=0.5, value=0.0, label=schemas.Label(key="name", value="dog")
        ),
        schemas.APMetric(
            iou=0.5,
            value=0.25,
            label=schemas.Label(key="name", value="person"),
        ),
        schemas.APMetric(
            iou=0.75, value=0.44, label=schemas.Label(key="name", value="car")
        ),
        schemas.APMetric(
            iou=0.75, value=0.0, label=schemas.Label(key="name", value="dog")
        ),
        schemas.APMetric(
            iou=0.75,
            value=0.25,
            label=schemas.Label(key="name", value="person"),
        ),
        schemas.APMetric(
            iou=0.9, value=0.11, label=schemas.Label(key="name", value="car")
        ),
        schemas.APMetric(
            iou=0.9, value=0.0, label=schemas.Label(key="name", value="dog")
        ),
        schemas.APMetric(
            iou=0.9,
            value=0.25,
            label=schemas.Label(key="name", value="person"),
        ),
    ]

    grouper_ids_associated_with_gts = set(["0", "1", "2"])

    detection_metrics = _ap(
        sorted_ranked_pairs=pairs,
        number_of_groundtruths_per_grouper=number_of_groundtruths_per_grouper,
        grouper_mappings=grouper_mappings,
        iou_thresholds=iou_thresholds,
        grouper_ids_associated_with_gts=grouper_ids_associated_with_gts,
    )

    assert len(reference_metrics) == len(detection_metrics)
    for pd, gt in zip(detection_metrics, reference_metrics):
        assert pd.iou == gt.iou
        assert truncate_float(pd.value) == truncate_float(gt.value)
        assert pd.label == gt.label
