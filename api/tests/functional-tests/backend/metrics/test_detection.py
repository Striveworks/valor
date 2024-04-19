import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import crud, enums, schemas
from valor_api.backend.metrics.detection import (
    RankedPair,
    _compute_curves,
    _compute_detection_metrics,
    compute_detection_metrics,
)
from valor_api.backend.models import (
    Dataset,
    Evaluation,
    GroundTruth,
    Model,
    Prediction,
)


def _round_dict(d: dict, prec: int = 3) -> None:
    """Modifies a dictionary in place by rounding every float in it
    to three decimal places
    """
    for k, v in d.items():
        if isinstance(v, float):
            d[k] = round(v, prec)
        elif isinstance(v, dict):
            _round_dict(v, prec)


def test__compute_curves(db: Session):
    # these inputs are taken directly from test__compute_detection_metrics (below)
    sorted_ranked_pairs = {
        -1519138795911397979: [
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid526",
                gt_geojson='{"type":"Polygon","coordinates":[[[277.11,103.84],[292.44,103.84],[292.44,150.72],[277.11,150.72],[277.11,103.84]]]}',
                gt_id=1340,
                pd_id=2389,
                score=0.953,
                iou=0.8775260257195348,
            ),
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid526",
                gt_geojson='{"type":"Polygon","coordinates":[[[462.08,105.09],[493.74,105.09],[493.74,146.99],[462.08,146.99],[462.08,105.09]]]}',
                gt_id=1339,
                pd_id=2388,
                score=0.805,
                iou=0.8811645870469409,
            ),
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid526",
                gt_geojson='{"type":"Polygon","coordinates":[[[326.94,97.05],[340.49,97.05],[340.49,122.98],[326.94,122.98],[326.94,97.05]]]}',
                gt_id=1337,
                pd_id=2386,
                score=0.611,
                iou=0.742765273311898,
            ),
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid526",
                gt_geojson='{"type":"Polygon","coordinates":[[[295.55,93.96],[313.97,93.96],[313.97,152.79],[295.55,152.79],[295.55,93.96]]]}',
                gt_id=1336,
                pd_id=2385,
                score=0.407,
                iou=0.8970133882595271,
            ),
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid526",
                gt_geojson='{"type":"Polygon","coordinates":[[[356.62,95.47],[372.33,95.47],[372.33,147.55],[356.62,147.55],[356.62,95.47]]]}',
                gt_id=1338,
                pd_id=2387,
                score=0.335,
                iou=1.0000000000000002,
            ),
        ],
        564624103770992353: [
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid527",
                gt_geojson='{"type":"Polygon","coordinates":[[[75.29,23.01],[91.85,23.01],[91.85,50.85],[75.29,50.85],[75.29,23.01]]]}',
                gt_id=1345,
                pd_id=2394,
                score=0.883,
                iou=0.9999999999999992,
            ),
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid527",
                gt_geojson='{"type":"Polygon","coordinates":[[[81.28,47.04],[98.66,47.04],[98.66,78.5],[81.28,78.5],[81.28,47.04]]]}',
                gt_id=1343,
                pd_id=2392,
                score=0.782,
                iou=0.8911860718171924,
            ),
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid527",
                gt_geojson='{"type":"Polygon","coordinates":[[[62.34,55.23],[78.14,55.23],[78.14,79.57],[62.34,79.57],[62.34,55.23]]]}',
                gt_id=1348,
                pd_id=2396,
                score=0.561,
                iou=0.8809523809523806,
            ),
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid527",
                gt_geojson='{"type":"Polygon","coordinates":[[[72.92,45.96],[91.23,45.96],[91.23,80.57],[72.92,80.57],[72.92,45.96]]]}',
                gt_id=1341,
                pd_id=2390,
                score=0.532,
                iou=0.9999999999999998,
            ),
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid527",
                gt_geojson='{"type":"Polygon","coordinates":[[[58.18,44.8],[66.42,44.8],[66.42,56.25],[58.18,56.25],[58.18,44.8]]]}',
                gt_id=1350,
                pd_id=2398,
                score=0.349,
                iou=0.6093750000000003,
            ),
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid527",
                gt_geojson='{"type":"Polygon","coordinates":[[[73.14,1.1],[98.96,1.1],[98.96,28.33],[73.14,28.33],[73.14,1.1]]]}',
                gt_id=1347,
                pd_id=2395,
                score=0.271,
                iou=0.8562185478073326,
            ),
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid527",
                gt_geojson='{"type":"Polygon","coordinates":[[[44.17,45.78],[63.99,45.78],[63.99,78.48],[44.17,78.48],[44.17,45.78]]]}',
                gt_id=1349,
                pd_id=2391,
                score=0.204,
                iou=0.8089209038203885,
            ),
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid527",
                gt_geojson='{"type":"Polygon","coordinates":[[[50.17,45.34],[71.28,45.34],[71.28,79.83],[50.17,79.83],[50.17,45.34]]]}',
                gt_id=1342,
                pd_id=2397,
                score=0.204,
                iou=0.3460676561905953,
            ),
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid527",
                gt_geojson='{"type":"Polygon","coordinates":[[[63.96,46.17],[84.35,46.17],[84.35,80.48],[63.96,80.48],[63.96,46.17]]]}',
                gt_id=1344,
                pd_id=2393,
                score=0.202,
                iou=0.6719967199671995,
            ),
        ],
        7641129594263252302: [
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid525",
                gt_geojson='{"type":"Polygon","coordinates":[[[1.66,3.32],[270.26,3.32],[270.26,275.23],[1.66,275.23],[1.66,3.32]]]}',
                gt_id=1333,
                pd_id=2382,
                score=0.726,
                iou=0.9213161659513592,
            )
        ],
        7594118964129415143: [
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid526",
                gt_geojson='{"type":"Polygon","coordinates":[[[61.87,276.25],[358.29,276.25],[358.29,379.43],[61.87,379.43],[61.87,276.25]]]}',
                gt_id=1334,
                pd_id=2383,
                score=0.546,
                iou=0.8387196824018363,
            ),
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid524",
                gt_geojson='{"type":"Polygon","coordinates":[[[214.15,41.29],[562.41,41.29],[562.41,285.07],[214.15,285.07],[214.15,41.29]]]}',
                gt_id=1331,
                pd_id=2380,
                score=0.236,
                iou=0.7756590016825575,
            ),
        ],
        8707070029533313719: [
            RankedPair(
                dataset_name="test_dataset",
                gt_datum_uid="uid526",
                gt_geojson='{"type":"Polygon","coordinates":[[[2.75,3.66],[162.15,3.66],[162.15,316.06],[2.75,316.06],[2.75,3.66]]]}',
                gt_id=1335,
                pd_id=2384,
                score=0.3,
                iou=0.8596978106691334,
            )
        ],
    }

    grouper_mappings = {
        "label_id_to_grouper_id_mapping": {
            752: 7594118964129415143,
            754: -1519138795911397979,
            753: 7641129594263252302,
            755: 8707070029533313719,
            757: 1005277842145977801,
            756: 564624103770992353,
        },
        "grouper_id_to_label_ids_mapping": {
            7594118964129415143: [752],
            -1519138795911397979: [754],
            7641129594263252302: [753],
            8707070029533313719: [755],
            1005277842145977801: [757],
            564624103770992353: [756],
        },
        "grouper_id_to_grouper_label_mapping": {
            7594118964129415143: schemas.Label(
                key="class", value="4", score=None
            ),
            -1519138795911397979: schemas.Label(
                key="class", value="0", score=None
            ),
            7641129594263252302: schemas.Label(
                key="class", value="2", score=None
            ),
            8707070029533313719: schemas.Label(
                key="class", value="1", score=None
            ),
            1005277842145977801: schemas.Label(
                key="class", value="3", score=None
            ),
            564624103770992353: schemas.Label(
                key="class", value="49", score=None
            ),
        },
    }
    groundtruths_per_grouper = {
        7594118964129415143: [
            (
                "test_dataset",
                "uid524",
                1331,
                '{"type":"Polygon","coordinates":[[[214.15,41.29],[562.41,41.29],[562.41,285.07],[214.15,285.07],[214.15,41.29]]]}',
            ),
            (
                "test_dataset",
                "uid526",
                1334,
                '{"type":"Polygon","coordinates":[[[61.87,276.25],[358.29,276.25],[358.29,379.43],[61.87,379.43],[61.87,276.25]]]}',
            ),
        ],
        7641129594263252302: [
            (
                "test_dataset",
                "uid525",
                1332,
                '{"type":"Polygon","coordinates":[[[13,22.75],[548.98,22.75],[548.98,632.42],[13,632.42],[13,22.75]]]}',
            ),
            (
                "test_dataset",
                "uid525",
                1333,
                '{"type":"Polygon","coordinates":[[[1.66,3.32],[270.26,3.32],[270.26,275.23],[1.66,275.23],[1.66,3.32]]]}',
            ),
        ],
        8707070029533313719: [
            (
                "test_dataset",
                "uid526",
                1335,
                '{"type":"Polygon","coordinates":[[[2.75,3.66],[162.15,3.66],[162.15,316.06],[2.75,316.06],[2.75,3.66]]]}',
            )
        ],
        -1519138795911397979: [
            (
                "test_dataset",
                "uid526",
                1336,
                '{"type":"Polygon","coordinates":[[[295.55,93.96],[313.97,93.96],[313.97,152.79],[295.55,152.79],[295.55,93.96]]]}',
            ),
            (
                "test_dataset",
                "uid526",
                1337,
                '{"type":"Polygon","coordinates":[[[326.94,97.05],[340.49,97.05],[340.49,122.98],[326.94,122.98],[326.94,97.05]]]}',
            ),
            (
                "test_dataset",
                "uid526",
                1338,
                '{"type":"Polygon","coordinates":[[[356.62,95.47],[372.33,95.47],[372.33,147.55],[356.62,147.55],[356.62,95.47]]]}',
            ),
            (
                "test_dataset",
                "uid526",
                1339,
                '{"type":"Polygon","coordinates":[[[462.08,105.09],[493.74,105.09],[493.74,146.99],[462.08,146.99],[462.08,105.09]]]}',
            ),
            (
                "test_dataset",
                "uid526",
                1340,
                '{"type":"Polygon","coordinates":[[[277.11,103.84],[292.44,103.84],[292.44,150.72],[277.11,150.72],[277.11,103.84]]]}',
            ),
        ],
        564624103770992353: [
            (
                "test_dataset",
                "uid527",
                1341,
                '{"type":"Polygon","coordinates":[[[72.92,45.96],[91.23,45.96],[91.23,80.57],[72.92,80.57],[72.92,45.96]]]}',
            ),
            (
                "test_dataset",
                "uid527",
                1342,
                '{"type":"Polygon","coordinates":[[[50.17,45.34],[71.28,45.34],[71.28,79.83],[50.17,79.83],[50.17,45.34]]]}',
            ),
            (
                "test_dataset",
                "uid527",
                1343,
                '{"type":"Polygon","coordinates":[[[81.28,47.04],[98.66,47.04],[98.66,78.5],[81.28,78.5],[81.28,47.04]]]}',
            ),
            (
                "test_dataset",
                "uid527",
                1344,
                '{"type":"Polygon","coordinates":[[[63.96,46.17],[84.35,46.17],[84.35,80.48],[63.96,80.48],[63.96,46.17]]]}',
            ),
            (
                "test_dataset",
                "uid527",
                1345,
                '{"type":"Polygon","coordinates":[[[75.29,23.01],[91.85,23.01],[91.85,50.85],[75.29,50.85],[75.29,23.01]]]}',
            ),
            (
                "test_dataset",
                "uid527",
                1346,
                '{"type":"Polygon","coordinates":[[[56.39,21.65],[75.66,21.65],[75.66,45.54],[56.39,45.54],[56.39,21.65]]]}',
            ),
            (
                "test_dataset",
                "uid527",
                1347,
                '{"type":"Polygon","coordinates":[[[73.14,1.1],[98.96,1.1],[98.96,28.33],[73.14,28.33],[73.14,1.1]]]}',
            ),
            (
                "test_dataset",
                "uid527",
                1348,
                '{"type":"Polygon","coordinates":[[[62.34,55.23],[78.14,55.23],[78.14,79.57],[62.34,79.57],[62.34,55.23]]]}',
            ),
            (
                "test_dataset",
                "uid527",
                1349,
                '{"type":"Polygon","coordinates":[[[44.17,45.78],[63.99,45.78],[63.99,78.48],[44.17,78.48],[44.17,45.78]]]}',
            ),
            (
                "test_dataset",
                "uid527",
                1350,
                '{"type":"Polygon","coordinates":[[[58.18,44.8],[66.42,44.8],[66.42,56.25],[58.18,56.25],[58.18,44.8]]]}',
            ),
        ],
    }

    false_positive_entries = [
        (
            "test_dataset",
            None,
            "uid525",
            None,
            1005277842145977801,
            0.318,
            '{"type":"Polygon","coordinates":[[[61,22.75],[565,22.75],[565,632.42],[61,632.42],[61,22.75]]]}',
        )
    ]

    output = _compute_curves(
        sorted_ranked_pairs=sorted_ranked_pairs,
        grouper_mappings=grouper_mappings,
        groundtruths_per_grouper=groundtruths_per_grouper,
        false_positive_entries=false_positive_entries,
        iou_threshold=0.5,
    )

    pr_expected_answers = {
        # (class, 4)
        ("class", "4", 0.05, "tp"): 2,
        ("class", "4", 0.05, "fn"): 0,
        ("class", "4", 0.25, "tp"): 1,
        ("class", "4", 0.25, "fn"): 1,
        ("class", "4", 0.55, "tp"): 0,
        ("class", "4", 0.55, "fn"): 2,
        # (class, 2)
        ("class", "2", 0.05, "tp"): 1,
        ("class", "2", 0.05, "fn"): 1,
        ("class", "2", 0.75, "tp"): 0,
        ("class", "2", 0.75, "fn"): 2,
        # (class, 49)
        ("class", "49", 0.05, "tp"): 8,
        ("class", "49", 0.3, "tp"): 5,
        ("class", "49", 0.5, "tp"): 4,
        ("class", "49", 0.85, "tp"): 1,
        # (class, 3)
        ("class", "3", 0.05, "tp"): 0,
        ("class", "3", 0.05, "fp"): 1,
        # (class, 1)
        ("class", "1", 0.05, "tp"): 1,
        ("class", "1", 0.35, "tp"): 0,
        # (class, 0)
        ("class", "0", 0.05, "tp"): 5,
        ("class", "0", 0.5, "tp"): 3,
        ("class", "0", 0.95, "tp"): 1,
        ("class", "0", 0.95, "fn"): 4,
    }

    for (
        key,
        value,
        threshold,
        metric,
    ), expected_length in pr_expected_answers.items():
        datum_geojson_tuples = output[0].value[value][threshold][metric]
        assert isinstance(datum_geojson_tuples, list)
        assert len(datum_geojson_tuples) == expected_length

    # spot check a few geojson results
    assert (
        output[0].value["4"][0.05]["tp"][0][2]  # type: ignore
        == '{"type":"Polygon","coordinates":[[[61.87,276.25],[358.29,276.25],[358.29,379.43],[61.87,379.43],[61.87,276.25]]]}'
    )
    assert (
        output[0].value["49"][0.85]["tp"][0][2]  # type: ignore
        == '{"type":"Polygon","coordinates":[[[75.29,23.01],[91.85,23.01],[91.85,50.85],[75.29,50.85],[75.29,23.01]]]}'
    )
    assert (
        output[0].value["3"][0.05]["fp"][0][2]  # type: ignore
        == '{"type":"Polygon","coordinates":[[[61,22.75],[565,22.75],[565,632.42],[61,632.42],[61,22.75]]]}'
    )

    # do a second test with a much higher iou_threshold
    second_output = _compute_curves(
        sorted_ranked_pairs=sorted_ranked_pairs,
        grouper_mappings=grouper_mappings,
        groundtruths_per_grouper=groundtruths_per_grouper,
        false_positive_entries=false_positive_entries,
        iou_threshold=0.9,
    )

    pr_expected_answers = {
        # (class, 4)
        ("class", "4", 0.05, "tp"): 0,
        ("class", "4", 0.05, "fn"): 2,
        # (class, 2)
        ("class", "2", 0.05, "tp"): 1,
        ("class", "2", 0.05, "fn"): 1,
        ("class", "2", 0.75, "tp"): 0,
        ("class", "2", 0.75, "fn"): 2,
        # (class, 49)
        ("class", "49", 0.05, "tp"): 2,
        ("class", "49", 0.3, "tp"): 2,
        ("class", "49", 0.5, "tp"): 2,
        ("class", "49", 0.85, "tp"): 1,
        # (class, 3)
        ("class", "3", 0.05, "tp"): 0,
        ("class", "3", 0.05, "fp"): 1,
        # (class, 1)
        ("class", "1", 0.05, "tp"): 0,
        ("class", "1", 0.05, "fn"): 1,
        # (class, 0)
        ("class", "0", 0.05, "tp"): 1,
        ("class", "0", 0.5, "tp"): 0,
        ("class", "0", 0.95, "fn"): 5,
    }

    for (
        key,
        value,
        threshold,
        metric,
    ), expected_length in pr_expected_answers.items():
        datum_geojson_tuples = second_output[0].value[value][threshold][metric]
        assert isinstance(datum_geojson_tuples, list)
        assert len(datum_geojson_tuples) == expected_length


def test__compute_detection_metrics(
    db: Session,
    groundtruths: list[list[GroundTruth]],
    predictions: list[list[Prediction]],
):
    iou_thresholds = set([round(0.5 + 0.05 * i, 2) for i in range(10)])
    metrics = _compute_detection_metrics(
        db=db,
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
            convert_annotations_to_type=enums.AnnotationType.BOX,
            iou_thresholds_to_compute=list(iou_thresholds),
            iou_thresholds_to_return=[0.5, 0.75],
            compute_pr_curves=True,
        ),
        prediction_filter=schemas.Filter(
            model_names=["test_model"],
            label_keys=["class"],
        ),
        groundtruth_filter=schemas.Filter(
            dataset_names=["test_dataset"],
            label_keys=["class"],
        ),
        target_type=enums.AnnotationType.BOX,
    )

    def _metric_to_dict(m) -> dict:
        m = m.model_dump(exclude_none=True)
        _round_dict(m, 3)
        return m

    ap_metrics = [
        _metric_to_dict(m) for m in metrics if isinstance(m, schemas.APMetric)
    ]
    map_metrics = [
        _metric_to_dict(m) for m in metrics if isinstance(m, schemas.mAPMetric)
    ]
    ap_metrics_ave_over_ious = [
        _metric_to_dict(m)
        for m in metrics
        if isinstance(m, schemas.APMetricAveragedOverIOUs)
    ]
    map_metrics_ave_over_ious = [
        _metric_to_dict(m)
        for m in metrics
        if isinstance(m, schemas.mAPMetricAveragedOverIOUs)
    ]
    ar_metrics = [
        _metric_to_dict(m) for m in metrics if isinstance(m, schemas.ARMetric)
    ]
    mar_metrics = [
        _metric_to_dict(m) for m in metrics if isinstance(m, schemas.mARMetric)
    ]

    # cf with torch metrics/pycocotools results listed here:
    # https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    expected_ap_metrics = [
        {"iou": 0.5, "value": 0.505, "label": {"key": "class", "value": "2"}},
        {"iou": 0.75, "value": 0.505, "label": {"key": "class", "value": "2"}},
        {"iou": 0.5, "value": 0.79, "label": {"key": "class", "value": "49"}},
        {
            "iou": 0.75,
            "value": 0.576,
            "label": {"key": "class", "value": "49"},
        },
        {"iou": 0.5, "value": 1.0, "label": {"key": "class", "value": "0"}},
        {"iou": 0.75, "value": 0.723, "label": {"key": "class", "value": "0"}},
        {"iou": 0.5, "value": 1.0, "label": {"key": "class", "value": "1"}},
        {"iou": 0.75, "value": 1.0, "label": {"key": "class", "value": "1"}},
        {"iou": 0.5, "value": 1.0, "label": {"key": "class", "value": "4"}},
        {"iou": 0.75, "value": 1.0, "label": {"key": "class", "value": "4"}},
    ]
    expected_map_metrics = [
        {"iou": 0.5, "value": 0.859},
        {"iou": 0.75, "value": 0.761},
    ]
    expected_ap_metrics_ave_over_ious = [
        {
            "ious": iou_thresholds,
            "value": 0.454,
            "label": {"key": "class", "value": "2"},
        },
        {
            "ious": iou_thresholds,
            "value": 0.555,  # note COCO had 0.556
            "label": {"key": "class", "value": "49"},
        },
        {
            "ious": iou_thresholds,
            "value": 0.725,
            "label": {"key": "class", "value": "0"},
        },
        {
            "ious": iou_thresholds,
            "value": 0.8,
            "label": {"key": "class", "value": "1"},
        },
        {
            "ious": iou_thresholds,
            "value": 0.650,
            "label": {"key": "class", "value": "4"},
        },
    ]
    expected_map_metrics_ave_over_ious = [
        {"ious": iou_thresholds, "value": 0.637}
    ]
    expected_ar_metrics = [
        {
            "ious": iou_thresholds,
            "value": 0.45,
            "label": {"key": "class", "value": "2"},
        },
        {
            "ious": iou_thresholds,
            "value": -1,
            "label": {"key": "class", "value": "3"},
        },
        {
            "ious": iou_thresholds,
            "value": 0.58,
            "label": {"key": "class", "value": "49"},
        },
        {
            "ious": iou_thresholds,
            "value": 0.78,
            "label": {"key": "class", "value": "0"},
        },
        {
            "ious": iou_thresholds,
            "value": 0.8,
            "label": {"key": "class", "value": "1"},
        },
        {
            "ious": iou_thresholds,
            "value": 0.65,
            "label": {"key": "class", "value": "4"},
        },
    ]
    expected_mar_metrics = [
        {
            "ious": iou_thresholds,
            "value": 0.652,
        },
    ]

    for metric_type, actual_metrics, expected_metrics in [
        ("AP", ap_metrics, expected_ap_metrics),
        ("mAP", map_metrics, expected_map_metrics),
        (
            "APAveOverIOUs",
            ap_metrics_ave_over_ious,
            expected_ap_metrics_ave_over_ious,
        ),
        (
            "mAPAveOverIOUs",
            map_metrics_ave_over_ious,
            expected_map_metrics_ave_over_ious,
        ),
        ("AR", ar_metrics, expected_ar_metrics),
        ("mAR", mar_metrics, expected_mar_metrics),
    ]:

        for m in actual_metrics:
            assert m in expected_metrics, f"{metric_type} {m} not in expected"
        for m in expected_metrics:
            assert m in actual_metrics, f"{metric_type} {m} not in actual"

    pr_metrics = metrics[-1].model_dump(exclude_none=True)

    pr_expected_answers = {
        # (class, 4)
        ("class", "4", 0.05, "tp"): 2,
        ("class", "4", 0.05, "fn"): 0,
        ("class", "4", 0.25, "tp"): 1,
        ("class", "4", 0.25, "fn"): 1,
        ("class", "4", 0.55, "tp"): 0,
        ("class", "4", 0.55, "fn"): 2,
        # (class, 2)
        ("class", "2", 0.05, "tp"): 1,
        ("class", "2", 0.05, "fn"): 1,
        ("class", "2", 0.75, "tp"): 0,
        ("class", "2", 0.75, "fn"): 2,
        # (class, 49)
        ("class", "49", 0.05, "tp"): 8,
        ("class", "49", 0.3, "tp"): 5,
        ("class", "49", 0.5, "tp"): 4,
        ("class", "49", 0.85, "tp"): 1,
        # (class, 3)
        ("class", "3", 0.05, "tp"): 0,
        ("class", "3", 0.05, "fp"): 1,
        # (class, 1)
        ("class", "1", 0.05, "tp"): 1,
        ("class", "1", 0.35, "tp"): 0,
        # (class, 0)
        ("class", "0", 0.05, "tp"): 5,
        ("class", "0", 0.5, "tp"): 3,
        ("class", "0", 0.95, "tp"): 1,
        ("class", "0", 0.95, "fn"): 4,
    }

    for (
        _,
        value,
        threshold,
        metric,
    ), expected_length in pr_expected_answers.items():
        assert (
            len(pr_metrics["value"][value][threshold][metric])
            == expected_length
        )

    # spot check a few geojson results
    assert (
        pr_metrics["value"]["4"][0.05]["tp"][0][2]
        == '{"type":"Polygon","coordinates":[[[61.87,276.25],[358.29,276.25],[358.29,379.43],[61.87,379.43],[61.87,276.25]]]}'
    )
    assert (
        pr_metrics["value"]["49"][0.85]["tp"][0][2]
        == '{"type":"Polygon","coordinates":[[[75.29,23.01],[91.85,23.01],[91.85,50.85],[75.29,50.85],[75.29,23.01]]]}'
    )
    assert (
        pr_metrics["value"]["3"][0.05]["fp"][0][2]
        == '{"type":"Polygon","coordinates":[[[61,22.75],[565,22.75],[565,632.42],[61,632.42],[61,22.75]]]}'
    )


def test__compute_detection_metrics_with_rasters(
    db: Session,
    groundtruths_with_rasters: list[list[GroundTruth]],
    predictions_with_rasters: list[list[Prediction]],
):
    iou_thresholds = set([round(0.5 + 0.05 * i, 2) for i in range(10)])
    metrics = _compute_detection_metrics(
        db=db,
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
            convert_annotations_to_type=enums.AnnotationType.RASTER,
            iou_thresholds_to_compute=list(iou_thresholds),
            iou_thresholds_to_return=[0.5, 0.75],
            compute_pr_curves=True,
        ),
        prediction_filter=schemas.Filter(
            model_names=["test_model"],
            label_keys=["class"],
        ),
        groundtruth_filter=schemas.Filter(
            dataset_names=["test_dataset"],
            label_keys=["class"],
        ),
        target_type=enums.AnnotationType.RASTER,
    )

    metrics = [m.model_dump(exclude_none=True) for m in metrics]

    for m in metrics:
        _round_dict(m, 3)

    expected = [
        # AP METRICS
        {
            "iou": 0.5,
            "value": 1.0,
            "label": {"key": "class", "value": "label2"},
        },
        {
            "iou": 0.75,
            "value": 1.0,
            "label": {"key": "class", "value": "label2"},
        },
        {
            "iou": 0.5,
            "value": 1.0,
            "label": {"key": "class", "value": "label1"},
        },
        {
            "iou": 0.75,
            "value": 1.0,
            "label": {"key": "class", "value": "label1"},
        },
        {
            "iou": 0.5,
            "value": 0.0,
            "label": {"key": "class", "value": "label3"},
        },
        {
            "iou": 0.75,
            "value": 0.0,
            "label": {"key": "class", "value": "label3"},
        },
        # AP METRICS AVERAGED OVER IOUS
        {
            "ious": iou_thresholds,
            "value": 1.0,
            "label": {"key": "class", "value": "label2"},
        },
        {
            "ious": iou_thresholds,
            "value": -1.0,
            "label": {"key": "class", "value": "label4"},
        },
        {
            "ious": iou_thresholds,
            "value": 1.0,
            "label": {"key": "class", "value": "label1"},
        },
        {
            "ious": iou_thresholds,
            "value": 0.0,
            "label": {"key": "class", "value": "label3"},
        },
        # mAP METRICS
        {"iou": 0.5, "value": 0.667},
        {"iou": 0.75, "value": 0.667},
        # mAP METRICS AVERAGED OVER IOUS
        {
            "ious": iou_thresholds,
            "value": 0.667,
        },
        # AR METRICS
        {
            "ious": iou_thresholds,
            "value": 1.0,
            "label": {"key": "class", "value": "label2"},
        },
        {
            "ious": iou_thresholds,
            "value": 1.0,
            "label": {"key": "class", "value": "label1"},
        },
        {
            "ious": iou_thresholds,
            "value": 0.0,
            "label": {"key": "class", "value": "label3"},
        },
        # mAR METRICS
        {
            "ious": iou_thresholds,
            "value": 0.667,
        },
    ]

    non_pr_metrics = metrics[:-1]
    pr_metrics = metrics[-1]
    for m in non_pr_metrics:
        assert m in expected

    for m in expected:
        assert m in non_pr_metrics

    pr_expected_answers = {
        ("class", "label1", 0.05, "tp"): 1,
        ("class", "label1", 0.35, "tp"): 0,
        ("class", "label2", 0.05, "tp"): 1,
        ("class", "label2", 0.05, "fp"): 0,
        ("class", "label2", 0.95, "fp"): 0,
        ("class", "label3", 0.05, "tp"): 0,
        ("class", "label3", 0.05, "fn"): 1,
        ("class", "label4", 0.05, "tp"): 0,
        ("class", "label4", 0.05, "fp"): 1,
    }

    for (
        _,
        value,
        threshold,
        metric,
    ), expected_length in pr_expected_answers.items():
        assert (
            len(pr_metrics["value"][value][threshold][metric])
            == expected_length
        )

    # spot check a few geojson results
    assert (
        pr_metrics["value"]["label1"][0.05]["tp"][0][2]
        == '{"type":"Polygon","coordinates":[[[0,0],[0,80],[32,80],[32,0],[0,0]]]}'
    )
    assert (
        pr_metrics["value"]["label2"][0.85]["tp"][0][2]
        == '{"type":"Polygon","coordinates":[[[0,0],[0,80],[32,80],[32,0],[0,0]]]}'
    )

    assert pr_metrics["value"]["label3"][0.85]["tp"] == []


def test_detection_exceptions(db: Session):
    dataset_name = "myDataset1"
    model_name = "myModel1"

    dataset = Dataset(
        name=dataset_name,
        meta=dict(),
        status=enums.TableStatus.CREATING,
    )
    model = Model(
        name=model_name,
        meta=dict(),
        status=enums.ModelStatus.READY,
    )
    evaluation = Evaluation(
        model_name=model_name,
        datum_filter={"dataset_names": [dataset_name]},
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
            iou_thresholds_to_compute=[0.5],
            iou_thresholds_to_return=[0.5],
        ).model_dump(),
        status=enums.EvaluationStatus.PENDING,
        meta={},
    )
    try:
        db.add(dataset)
        db.add(model)
        db.add(evaluation)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
    row = db.query(Evaluation).one_or_none()
    assert row
    evaluation_id = row.id

    # test that no datasets are found that meet the filter requirements
    # - this is b/c no ground truths exist that match the evaluation task type.
    with pytest.raises(RuntimeError) as e:
        compute_detection_metrics(db=db, evaluation_id=evaluation_id)
    assert "No datasets could be found that meet filter requirements." in str(
        e
    )

    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            dataset_name=dataset_name,
            datum=schemas.Datum(uid="uid"),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.OBJECT_DETECTION,
                    labels=[schemas.Label(key="k1", value="v1")],
                    bounding_box=schemas.Box.from_extrema(
                        xmin=0, xmax=1, ymin=0, ymax=1
                    ),
                )
            ],
        ),
    )

    # test that the model does not meet the filter requirements
    # - this is b/c no predictions exist that match the evaluation task type.
    with pytest.raises(RuntimeError) as e:
        compute_detection_metrics(db=db, evaluation_id=evaluation_id)
    assert f"Model '{model_name}' does not meet filter requirements." in str(e)

    crud.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            dataset_name=dataset_name,
            model_name=model_name,
            datum=schemas.Datum(uid="uid"),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.OBJECT_DETECTION,
                    labels=[schemas.Label(key="k1", value="v1", score=1.0)],
                    bounding_box=schemas.Box.from_extrema(
                        xmin=0, xmax=1, ymin=0, ymax=1
                    ),
                )
            ],
        ),
    )

    # show that no errors raised
    compute_detection_metrics(db=db, evaluation_id=evaluation_id)
