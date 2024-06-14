import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import crud, enums, schemas
from valor_api.backend.metrics.detection import (
    RankedPair,
    _compute_detailed_curves,
    _compute_detection_metrics,
    _compute_detection_metrics_with_detailed_precision_recall_curve,
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


def test__compute_detailed_curves(db: Session):
    # these inputs are taken directly from test__compute_detection_metrics (below)
    sorted_ranked_pairs = {
        3262893736873277849: [
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="2",
                gt_datum_uid="2",
                gt_geojson='{"type":"Polygon","coordinates":[[[277.11,103.84],[292.44,103.84],[292.44,150.72],[277.11,150.72],[277.11,103.84]]]}',
                gt_id=404,
                pd_id=397,
                score=0.953,
                iou=0.8775260257195348,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="2",
                gt_datum_uid="2",
                gt_geojson='{"type":"Polygon","coordinates":[[[277.11,103.84],[292.44,103.84],[292.44,150.72],[277.11,150.72],[277.11,103.84]]]}',
                gt_id=404,
                pd_id=397,
                score=0.953,
                iou=0.8775260257195348,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="2",
                gt_datum_uid="2",
                gt_geojson='{"type":"Polygon","coordinates":[[[462.08,105.09],[493.74,105.09],[493.74,146.99],[462.08,146.99],[462.08,105.09]]]}',
                gt_id=403,
                pd_id=396,
                score=0.805,
                iou=0.8811645870469409,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="2",
                gt_datum_uid="2",
                gt_geojson='{"type":"Polygon","coordinates":[[[462.08,105.09],[493.74,105.09],[493.74,146.99],[462.08,146.99],[462.08,105.09]]]}',
                gt_id=403,
                pd_id=396,
                score=0.805,
                iou=0.8811645870469409,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="2",
                gt_datum_uid="2",
                gt_geojson='{"type":"Polygon","coordinates":[[[326.94,97.05],[340.49,97.05],[340.49,122.98],[326.94,122.98],[326.94,97.05]]]}',
                gt_id=401,
                pd_id=394,
                score=0.611,
                iou=0.742765273311898,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="2",
                gt_datum_uid="2",
                gt_geojson='{"type":"Polygon","coordinates":[[[326.94,97.05],[340.49,97.05],[340.49,122.98],[326.94,122.98],[326.94,97.05]]]}',
                gt_id=401,
                pd_id=394,
                score=0.611,
                iou=0.742765273311898,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="2",
                gt_datum_uid="2",
                gt_geojson='{"type":"Polygon","coordinates":[[[295.55,93.96],[313.97,93.96],[313.97,152.79],[295.55,152.79],[295.55,93.96]]]}',
                gt_id=400,
                pd_id=393,
                score=0.407,
                iou=0.8970133882595271,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="2",
                gt_datum_uid="2",
                gt_geojson='{"type":"Polygon","coordinates":[[[295.55,93.96],[313.97,93.96],[313.97,152.79],[295.55,152.79],[295.55,93.96]]]}',
                gt_id=400,
                pd_id=393,
                score=0.407,
                iou=0.8970133882595271,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="2",
                gt_datum_uid="2",
                gt_geojson='{"type":"Polygon","coordinates":[[[356.62,95.47],[372.33,95.47],[372.33,147.55],[356.62,147.55],[356.62,95.47]]]}',
                gt_id=402,
                pd_id=395,
                score=0.335,
                iou=1.0000000000000002,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="2",
                gt_datum_uid="2",
                gt_geojson='{"type":"Polygon","coordinates":[[[356.62,95.47],[372.33,95.47],[372.33,147.55],[356.62,147.55],[356.62,95.47]]]}',
                gt_id=402,
                pd_id=395,
                score=0.335,
                iou=1.0000000000000002,
                is_match=True,
            ),
        ],
        8850376905924579852: [
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[75.29,23.01],[91.85,23.01],[91.85,50.85],[75.29,50.85],[75.29,23.01]]]}',
                gt_id=409,
                pd_id=402,
                score=0.883,
                iou=0.9999999999999992,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[75.29,23.01],[91.85,23.01],[91.85,50.85],[75.29,50.85],[75.29,23.01]]]}',
                gt_id=409,
                pd_id=402,
                score=0.883,
                iou=0.9999999999999992,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[81.28,47.04],[98.66,47.04],[98.66,78.5],[81.28,78.5],[81.28,47.04]]]}',
                gt_id=407,
                pd_id=400,
                score=0.782,
                iou=0.8911860718171924,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[81.28,47.04],[98.66,47.04],[98.66,78.5],[81.28,78.5],[81.28,47.04]]]}',
                gt_id=407,
                pd_id=400,
                score=0.782,
                iou=0.8911860718171924,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[62.34,55.23],[78.14,55.23],[78.14,79.57],[62.34,79.57],[62.34,55.23]]]}',
                gt_id=412,
                pd_id=404,
                score=0.561,
                iou=0.8809523809523806,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[62.34,55.23],[78.14,55.23],[78.14,79.57],[62.34,79.57],[62.34,55.23]]]}',
                gt_id=412,
                pd_id=404,
                score=0.561,
                iou=0.8809523809523806,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[72.92,45.96],[91.23,45.96],[91.23,80.57],[72.92,80.57],[72.92,45.96]]]}',
                gt_id=405,
                pd_id=398,
                score=0.532,
                iou=0.9999999999999998,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[72.92,45.96],[91.23,45.96],[91.23,80.57],[72.92,80.57],[72.92,45.96]]]}',
                gt_id=405,
                pd_id=398,
                score=0.532,
                iou=0.9999999999999998,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[58.18,44.8],[66.42,44.8],[66.42,56.25],[58.18,56.25],[58.18,44.8]]]}',
                gt_id=414,
                pd_id=406,
                score=0.349,
                iou=0.6093750000000003,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[58.18,44.8],[66.42,44.8],[66.42,56.25],[58.18,56.25],[58.18,44.8]]]}',
                gt_id=414,
                pd_id=406,
                score=0.349,
                iou=0.6093750000000003,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[73.14,1.1],[98.96,1.1],[98.96,28.33],[73.14,28.33],[73.14,1.1]]]}',
                gt_id=411,
                pd_id=403,
                score=0.271,
                iou=0.8562185478073326,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[73.14,1.1],[98.96,1.1],[98.96,28.33],[73.14,28.33],[73.14,1.1]]]}',
                gt_id=411,
                pd_id=403,
                score=0.271,
                iou=0.8562185478073326,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[44.17,45.78],[63.99,45.78],[63.99,78.48],[44.17,78.48],[44.17,45.78]]]}',
                gt_id=413,
                pd_id=399,
                score=0.204,
                iou=0.8089209038203885,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[44.17,45.78],[63.99,45.78],[63.99,78.48],[44.17,78.48],[44.17,45.78]]]}',
                gt_id=413,
                pd_id=399,
                score=0.204,
                iou=0.8089209038203885,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[44.17,45.78],[63.99,45.78],[63.99,78.48],[44.17,78.48],[44.17,45.78]]]}',
                gt_id=413,
                pd_id=405,
                score=0.204,
                iou=0.7370727432077125,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[44.17,45.78],[63.99,45.78],[63.99,78.48],[44.17,78.48],[44.17,45.78]]]}',
                gt_id=413,
                pd_id=405,
                score=0.204,
                iou=0.7370727432077125,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[63.96,46.17],[84.35,46.17],[84.35,80.48],[63.96,80.48],[63.96,46.17]]]}',
                gt_id=408,
                pd_id=401,
                score=0.202,
                iou=0.6719967199671995,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="3",
                gt_datum_uid="3",
                gt_geojson='{"type":"Polygon","coordinates":[[[63.96,46.17],[84.35,46.17],[84.35,80.48],[63.96,80.48],[63.96,46.17]]]}',
                gt_id=408,
                pd_id=401,
                score=0.202,
                iou=0.6719967199671995,
                is_match=True,
            ),
        ],
        7683992730431173493: [
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="1",
                gt_datum_uid="1",
                gt_geojson='{"type":"Polygon","coordinates":[[[1.66,3.32],[270.26,3.32],[270.26,275.23],[1.66,275.23],[1.66,3.32]]]}',
                gt_id=397,
                pd_id=390,
                score=0.726,
                iou=0.9213161659513592,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="1",
                gt_datum_uid="1",
                gt_geojson='{"type":"Polygon","coordinates":[[[1.66,3.32],[270.26,3.32],[270.26,275.23],[1.66,275.23],[1.66,3.32]]]}',
                gt_id=397,
                pd_id=390,
                score=0.726,
                iou=0.9213161659513592,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="1",
                gt_datum_uid="1",
                gt_geojson='{"type":"Polygon","coordinates":[[[13,22.75],[548.98,22.75],[548.98,632.42],[13,632.42],[13,22.75]]]}',
                gt_id=396,
                pd_id=389,
                score=0.318,
                iou=0.8840217391304347,
                is_match=False,
            ),
        ],
        1591437737079826217: [
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="2",
                gt_datum_uid="2",
                gt_geojson='{"type":"Polygon","coordinates":[[[61.87,276.25],[358.29,276.25],[358.29,379.43],[61.87,379.43],[61.87,276.25]]]}',
                gt_id=398,
                pd_id=391,
                score=0.546,
                iou=0.8387196824018363,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="2",
                gt_datum_uid="2",
                gt_geojson='{"type":"Polygon","coordinates":[[[61.87,276.25],[358.29,276.25],[358.29,379.43],[61.87,379.43],[61.87,276.25]]]}',
                gt_id=398,
                pd_id=391,
                score=0.546,
                iou=0.8387196824018363,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="0",
                gt_datum_uid="0",
                gt_geojson='{"type":"Polygon","coordinates":[[[214.15,41.29],[562.41,41.29],[562.41,285.07],[214.15,285.07],[214.15,41.29]]]}',
                gt_id=395,
                pd_id=388,
                score=0.236,
                iou=0.7756590016825575,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="0",
                gt_datum_uid="0",
                gt_geojson='{"type":"Polygon","coordinates":[[[214.15,41.29],[562.41,41.29],[562.41,285.07],[214.15,285.07],[214.15,41.29]]]}',
                gt_id=395,
                pd_id=388,
                score=0.236,
                iou=0.7756590016825575,
                is_match=True,
            ),
        ],
        -487256420494681688: [
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="2",
                gt_datum_uid="2",
                gt_geojson='{"type":"Polygon","coordinates":[[[2.75,3.66],[162.15,3.66],[162.15,316.06],[2.75,316.06],[2.75,3.66]]]}',
                gt_id=399,
                pd_id=392,
                score=0.3,
                iou=0.8596978106691334,
                is_match=True,
            ),
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="2",
                gt_datum_uid="2",
                gt_geojson='{"type":"Polygon","coordinates":[[[2.75,3.66],[162.15,3.66],[162.15,316.06],[2.75,316.06],[2.75,3.66]]]}',
                gt_id=399,
                pd_id=392,
                score=0.3,
                iou=0.8596978106691334,
                is_match=True,
            ),
        ],
        -6111942735542320034: [
            RankedPair(
                dataset_name="test_dataset",
                pd_datum_uid="1",
                gt_datum_uid="1",
                gt_geojson='{"type":"Polygon","coordinates":[[[13,22.75],[548.98,22.75],[548.98,632.42],[13,632.42],[13,22.75]]]}',
                gt_id=396,
                pd_id=389,
                score=0.318,
                iou=0.8840217391304347,
                is_match=False,
            )
        ],
    }

    grouper_mappings = {
        "label_id_to_grouper_id_mapping": {
            512: 1591437737079826217,
            513: 7683992730431173493,
            519: -6111942735542320034,
            515: -487256420494681688,
            514: 3262893736873277849,
            517: 8850376905924579852,
        },
        "label_id_to_grouper_key_mapping": {
            512: "class",
            513: "class",
            519: "class",
            515: "class",
            514: "class",
            517: "class",
        },
        "grouper_id_to_label_ids_mapping": {
            1591437737079826217: [512],
            7683992730431173493: [513],
            -6111942735542320034: [519],
            -487256420494681688: [515],
            3262893736873277849: [514],
            8850376905924579852: [517],
        },
        "grouper_id_to_grouper_label_mapping": {
            1591437737079826217: schemas.Label(
                key="class", value="4", score=None
            ),
            7683992730431173493: schemas.Label(
                key="class", value="2", score=None
            ),
            -6111942735542320034: schemas.Label(
                key="class", value="3", score=None
            ),
            -487256420494681688: schemas.Label(
                key="class", value="1", score=None
            ),
            3262893736873277849: schemas.Label(
                key="class", value="0", score=None
            ),
            8850376905924579852: schemas.Label(
                key="class", value="49", score=None
            ),
        },
    }
    groundtruths_per_grouper = {
        1591437737079826217: [
            (
                "test_dataset",
                "0",
                395,
                '{"type":"Polygon","coordinates":[[[214.15,41.29],[562.41,41.29],[562.41,285.07],[214.15,285.07],[214.15,41.29]]]}',
            ),
            (
                "test_dataset",
                "2",
                398,
                '{"type":"Polygon","coordinates":[[[61.87,276.25],[358.29,276.25],[358.29,379.43],[61.87,379.43],[61.87,276.25]]]}',
            ),
        ],
        7683992730431173493: [
            (
                "test_dataset",
                "1",
                396,
                '{"type":"Polygon","coordinates":[[[13,22.75],[548.98,22.75],[548.98,632.42],[13,632.42],[13,22.75]]]}',
            ),
            (
                "test_dataset",
                "1",
                397,
                '{"type":"Polygon","coordinates":[[[1.66,3.32],[270.26,3.32],[270.26,275.23],[1.66,275.23],[1.66,3.32]]]}',
            ),
        ],
        -487256420494681688: [
            (
                "test_dataset",
                "2",
                399,
                '{"type":"Polygon","coordinates":[[[2.75,3.66],[162.15,3.66],[162.15,316.06],[2.75,316.06],[2.75,3.66]]]}',
            )
        ],
        3262893736873277849: [
            (
                "test_dataset",
                "2",
                400,
                '{"type":"Polygon","coordinates":[[[295.55,93.96],[313.97,93.96],[313.97,152.79],[295.55,152.79],[295.55,93.96]]]}',
            ),
            (
                "test_dataset",
                "2",
                401,
                '{"type":"Polygon","coordinates":[[[326.94,97.05],[340.49,97.05],[340.49,122.98],[326.94,122.98],[326.94,97.05]]]}',
            ),
            (
                "test_dataset",
                "2",
                402,
                '{"type":"Polygon","coordinates":[[[356.62,95.47],[372.33,95.47],[372.33,147.55],[356.62,147.55],[356.62,95.47]]]}',
            ),
            (
                "test_dataset",
                "2",
                403,
                '{"type":"Polygon","coordinates":[[[462.08,105.09],[493.74,105.09],[493.74,146.99],[462.08,146.99],[462.08,105.09]]]}',
            ),
            (
                "test_dataset",
                "2",
                404,
                '{"type":"Polygon","coordinates":[[[277.11,103.84],[292.44,103.84],[292.44,150.72],[277.11,150.72],[277.11,103.84]]]}',
            ),
        ],
        8850376905924579852: [
            (
                "test_dataset",
                "3",
                405,
                '{"type":"Polygon","coordinates":[[[72.92,45.96],[91.23,45.96],[91.23,80.57],[72.92,80.57],[72.92,45.96]]]}',
            ),
            (
                "test_dataset",
                "3",
                406,
                '{"type":"Polygon","coordinates":[[[50.17,45.34],[71.28,45.34],[71.28,79.83],[50.17,79.83],[50.17,45.34]]]}',
            ),
            (
                "test_dataset",
                "3",
                407,
                '{"type":"Polygon","coordinates":[[[81.28,47.04],[98.66,47.04],[98.66,78.5],[81.28,78.5],[81.28,47.04]]]}',
            ),
            (
                "test_dataset",
                "3",
                408,
                '{"type":"Polygon","coordinates":[[[63.96,46.17],[84.35,46.17],[84.35,80.48],[63.96,80.48],[63.96,46.17]]]}',
            ),
            (
                "test_dataset",
                "3",
                409,
                '{"type":"Polygon","coordinates":[[[75.29,23.01],[91.85,23.01],[91.85,50.85],[75.29,50.85],[75.29,23.01]]]}',
            ),
            (
                "test_dataset",
                "3",
                410,
                '{"type":"Polygon","coordinates":[[[56.39,21.65],[75.66,21.65],[75.66,45.54],[56.39,45.54],[56.39,21.65]]]}',
            ),
            (
                "test_dataset",
                "3",
                411,
                '{"type":"Polygon","coordinates":[[[73.14,1.1],[98.96,1.1],[98.96,28.33],[73.14,28.33],[73.14,1.1]]]}',
            ),
            (
                "test_dataset",
                "3",
                412,
                '{"type":"Polygon","coordinates":[[[62.34,55.23],[78.14,55.23],[78.14,79.57],[62.34,79.57],[62.34,55.23]]]}',
            ),
            (
                "test_dataset",
                "3",
                413,
                '{"type":"Polygon","coordinates":[[[44.17,45.78],[63.99,45.78],[63.99,78.48],[44.17,78.48],[44.17,45.78]]]}',
            ),
            (
                "test_dataset",
                "3",
                414,
                '{"type":"Polygon","coordinates":[[[58.18,44.8],[66.42,44.8],[66.42,56.25],[58.18,56.25],[58.18,44.8]]]}',
            ),
        ],
    }
    predictions_per_grouper = {
        1591437737079826217: [
            (
                "test_dataset",
                "0",
                388,
                '{"type":"Polygon","coordinates":[[[258.15,41.29],[606.41,41.29],[606.41,285.07],[258.15,285.07],[258.15,41.29]]]}',
            ),
            (
                "test_dataset",
                "2",
                391,
                '{"type":"Polygon","coordinates":[[[87.87,276.25],[384.29,276.25],[384.29,379.43],[87.87,379.43],[87.87,276.25]]]}',
            ),
        ],
        -6111942735542320034: [
            (
                "test_dataset",
                "1",
                389,
                '{"type":"Polygon","coordinates":[[[61,22.75],[565,22.75],[565,632.42],[61,632.42],[61,22.75]]]}',
            )
        ],
        7683992730431173493: [
            (
                "test_dataset",
                "1",
                390,
                '{"type":"Polygon","coordinates":[[[12.66,3.32],[281.26,3.32],[281.26,275.23],[12.66,275.23],[12.66,3.32]]]}',
            )
        ],
        -487256420494681688: [
            (
                "test_dataset",
                "2",
                392,
                '{"type":"Polygon","coordinates":[[[0,3.66],[142.15,3.66],[142.15,316.06],[0,316.06],[0,3.66]]]}',
            )
        ],
        3262893736873277849: [
            (
                "test_dataset",
                "2",
                393,
                '{"type":"Polygon","coordinates":[[[296.55,93.96],[314.97,93.96],[314.97,152.79],[296.55,152.79],[296.55,93.96]]]}',
            ),
            (
                "test_dataset",
                "2",
                394,
                '{"type":"Polygon","coordinates":[[[328.94,97.05],[342.49,97.05],[342.49,122.98],[328.94,122.98],[328.94,97.05]]]}',
            ),
            (
                "test_dataset",
                "2",
                395,
                '{"type":"Polygon","coordinates":[[[356.62,95.47],[372.33,95.47],[372.33,147.55],[356.62,147.55],[356.62,95.47]]]}',
            ),
            (
                "test_dataset",
                "2",
                396,
                '{"type":"Polygon","coordinates":[[[464.08,105.09],[495.74,105.09],[495.74,146.99],[464.08,146.99],[464.08,105.09]]]}',
            ),
            (
                "test_dataset",
                "2",
                397,
                '{"type":"Polygon","coordinates":[[[276.11,103.84],[291.44,103.84],[291.44,150.72],[276.11,150.72],[276.11,103.84]]]}',
            ),
        ],
        8850376905924579852: [
            (
                "test_dataset",
                "3",
                398,
                '{"type":"Polygon","coordinates":[[[72.92,45.96],[91.23,45.96],[91.23,80.57],[72.92,80.57],[72.92,45.96]]]}',
            ),
            (
                "test_dataset",
                "3",
                399,
                '{"type":"Polygon","coordinates":[[[45.17,45.34],[66.28,45.34],[66.28,79.83],[45.17,79.83],[45.17,45.34]]]}',
            ),
            (
                "test_dataset",
                "3",
                400,
                '{"type":"Polygon","coordinates":[[[82.28,47.04],[99.66,47.04],[99.66,78.5],[82.28,78.5],[82.28,47.04]]]}',
            ),
            (
                "test_dataset",
                "3",
                401,
                '{"type":"Polygon","coordinates":[[[59.96,46.17],[80.35,46.17],[80.35,80.48],[59.96,80.48],[59.96,46.17]]]}',
            ),
            (
                "test_dataset",
                "3",
                402,
                '{"type":"Polygon","coordinates":[[[75.29,23.01],[91.85,23.01],[91.85,50.85],[75.29,50.85],[75.29,23.01]]]}',
            ),
            (
                "test_dataset",
                "3",
                403,
                '{"type":"Polygon","coordinates":[[[71.14,1.1],[96.96,1.1],[96.96,28.33],[71.14,28.33],[71.14,1.1]]]}',
            ),
            (
                "test_dataset",
                "3",
                404,
                '{"type":"Polygon","coordinates":[[[61.34,55.23],[77.14,55.23],[77.14,79.57],[61.34,79.57],[61.34,55.23]]]}',
            ),
            (
                "test_dataset",
                "3",
                405,
                '{"type":"Polygon","coordinates":[[[41.17,45.78],[60.99,45.78],[60.99,78.48],[41.17,78.48],[41.17,45.78]]]}',
            ),
            (
                "test_dataset",
                "3",
                406,
                '{"type":"Polygon","coordinates":[[[56.18,44.8],[64.42,44.8],[64.42,56.25],[56.18,56.25],[56.18,44.8]]]}',
            ),
        ],
    }

    output = _compute_detailed_curves(
        sorted_ranked_pairs=sorted_ranked_pairs,
        grouper_mappings=grouper_mappings,
        groundtruths_per_grouper=groundtruths_per_grouper,
        predictions_per_grouper=predictions_per_grouper,
        pr_curve_iou_threshold=0.5,
        pr_curve_max_examples=1,
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
    ), expected_count in pr_expected_answers.items():
        actual_count = output[0].value[value][threshold][metric]
        assert actual_count == expected_count

    # check DetailedPrecisionRecallCurve
    detailed_pr_expected_answers = {
        # (class, 4)
        ("4", 0.05, "tp"): {"all": 2, "total": 2},
        ("4", 0.05, "fn"): {
            "missed_detections": 0,
            "misclassifications": 0,
            "total": 0,
        },
        # (class, 2)
        ("2", 0.05, "tp"): {"all": 1, "total": 1},
        ("2", 0.05, "fn"): {
            "missed_detections": 0,
            "misclassifications": 1,
            "total": 1,
        },
        ("2", 0.75, "tp"): {"all": 0, "total": 0},
        ("2", 0.75, "fn"): {
            "missed_detections": 2,
            "misclassifications": 0,
            "total": 2,
        },
        # (class, 49)
        ("49", 0.05, "tp"): {"all": 8, "total": 8},
        # (class, 3)
        ("3", 0.05, "tp"): {"all": 0, "total": 0},
        ("3", 0.05, "fp"): {
            "hallucinations": 0,
            "misclassifications": 1,
            "total": 1,
        },
        # (class, 1)
        ("1", 0.05, "tp"): {"all": 1, "total": 1},
        ("1", 0.8, "fn"): {
            "missed_detections": 1,
            "misclassifications": 0,
            "total": 1,
        },
        # (class, 0)
        ("0", 0.05, "tp"): {"all": 5, "total": 5},
        ("0", 0.95, "fn"): {
            "missed_detections": 4,
            "misclassifications": 0,
            "total": 4,
        },
    }

    for (
        value,
        threshold,
        metric,
    ), expected_output in detailed_pr_expected_answers.items():
        model_output = output[1].value[value][threshold][metric]
        assert isinstance(model_output, dict)
        assert model_output["total"] == expected_output["total"]
        assert all(
            [
                model_output["observations"][key]["count"]  # type: ignore - we know this element is a dict
                == expected_output[key]
                for key in [
                    key
                    for key in expected_output.keys()
                    if key not in ["total"]
                ]
            ]
        )

    # spot check number of examples
    assert (
        len(
            output[1].value["0"][0.95]["fn"]["observations"]["missed_detections"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 1
    )
    assert (
        len(
            output[1].value["49"][0.05]["tp"]["observations"]["all"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 1
    )

    # do a second test with a much higher iou_threshold
    second_output = _compute_detailed_curves(
        sorted_ranked_pairs=sorted_ranked_pairs,
        grouper_mappings=grouper_mappings,
        groundtruths_per_grouper=groundtruths_per_grouper,
        predictions_per_grouper=predictions_per_grouper,
        pr_curve_iou_threshold=0.9,
        pr_curve_max_examples=1,
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
    ), expected_count in pr_expected_answers.items():
        actual_count = second_output[0].value[value][threshold][metric]
        assert actual_count == expected_count

    # check DetailedPrecisionRecallCurve
    detailed_pr_expected_answers = {
        # (class, 4)
        ("4", 0.05, "tp"): {"all": 0, "total": 0},
        ("4", 0.05, "fn"): {
            "missed_detections": 2,  # below IOU threshold of .9
            "misclassifications": 0,
            "total": 2,
        },
        # (class, 2)
        ("2", 0.05, "tp"): {"all": 1, "total": 1},
        ("2", 0.05, "fn"): {
            "missed_detections": 1,
            "misclassifications": 0,
            "total": 1,
        },
        ("2", 0.75, "tp"): {"all": 0, "total": 0},
        ("2", 0.75, "fn"): {
            "missed_detections": 2,
            "misclassifications": 0,
            "total": 2,
        },
        # (class, 49)
        ("49", 0.05, "tp"): {"all": 2, "total": 2},
        # (class, 3)
        ("3", 0.05, "tp"): {"all": 0, "total": 0},
        ("3", 0.05, "fp"): {
            "hallucinations": 1,
            "misclassifications": 0,
            "total": 1,
        },
        # (class, 1)
        ("1", 0.05, "tp"): {"all": 0, "total": 0},
        ("1", 0.8, "fn"): {
            "missed_detections": 1,
            "misclassifications": 0,
            "total": 1,
        },
        # (class, 0)
        ("0", 0.05, "tp"): {"all": 1, "total": 1},
        ("0", 0.95, "fn"): {
            "missed_detections": 5,
            "misclassifications": 0,
            "total": 5,
        },
    }

    for (
        value,
        threshold,
        metric,
    ), expected_output in detailed_pr_expected_answers.items():
        model_output = second_output[1].value[value][threshold][metric]
        assert isinstance(model_output, dict)
        assert model_output["total"] == expected_output["total"]
        assert all(
            [
                model_output["observations"][key]["count"]  # type: ignore - we know this element is a dict
                == expected_output[key]
                for key in [
                    key
                    for key in expected_output.keys()
                    if key not in ["total"]
                ]
            ]
        )

    # spot check number of examples
    assert (
        len(
            second_output[1].value["0"][0.95]["fn"]["observations"]["missed_detections"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 1
    )
    assert (
        len(
            second_output[1].value["49"][0.05]["tp"]["observations"]["all"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 1
    )

    # repeat the above, but with a higher pr_max_curves_example
    second_output = _compute_detailed_curves(
        sorted_ranked_pairs=sorted_ranked_pairs,
        grouper_mappings=grouper_mappings,
        groundtruths_per_grouper=groundtruths_per_grouper,
        predictions_per_grouper=predictions_per_grouper,
        pr_curve_iou_threshold=0.9,
        pr_curve_max_examples=3,
    )

    for (
        value,
        threshold,
        metric,
    ), expected_output in detailed_pr_expected_answers.items():
        model_output = second_output[1].value[value][threshold][metric]
        assert isinstance(model_output, dict)
        assert model_output["total"] == expected_output["total"]
        assert all(
            [
                model_output["observations"][key]["count"]  # type: ignore - we know this element is a dict
                == expected_output[key]
                for key in [
                    key
                    for key in expected_output.keys()
                    if key not in ["total"]
                ]
            ]
        )

    # spot check number of examples
    assert (
        len(
            second_output[1].value["0"][0.95]["fn"]["observations"]["missed_detections"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 3
    )
    assert (
        len(
            second_output[1].value["49"][0.05]["tp"]["observations"]["all"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 2
    )

    # test behavior if pr_curve_max_examples == 0
    second_output = _compute_detailed_curves(
        sorted_ranked_pairs=sorted_ranked_pairs,
        grouper_mappings=grouper_mappings,
        groundtruths_per_grouper=groundtruths_per_grouper,
        predictions_per_grouper=predictions_per_grouper,
        pr_curve_iou_threshold=0.9,
        pr_curve_max_examples=0,
    )

    for (
        value,
        threshold,
        metric,
    ), expected_output in detailed_pr_expected_answers.items():
        model_output = second_output[1].value[value][threshold][metric]
        assert isinstance(model_output, dict)
        assert model_output["total"] == expected_output["total"]
        assert all(
            [
                model_output["observations"][key]["count"]  # type: ignore - we know this element is a dict
                == expected_output[key]
                for key in [
                    key
                    for key in expected_output.keys()
                    if key not in ["total"]
                ]
            ]
        )

    # spot check number of examples
    assert (
        len(
            second_output[1].value["0"][0.95]["fn"]["observations"]["missed_detections"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 0
    )
    assert (
        len(
            second_output[1].value["49"][0.05]["tp"]["observations"]["all"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 0
    )


def test__compute_detection(
    db: Session,
    groundtruths: list[list[GroundTruth]],
    predictions: list[list[Prediction]],
):
    iou_thresholds = set([round(0.5 + 0.05 * i, 2) for i in range(10)])

    def _metric_to_dict(m) -> dict:
        m = m.model_dump(exclude_none=True)
        _round_dict(m, 3)
        return m

    metrics = _compute_detection_metrics(
        db=db,
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
            convert_annotations_to_type=enums.AnnotationType.BOX,
            iou_thresholds_to_compute=list(iou_thresholds),
            iou_thresholds_to_return=[0.5, 0.75],
            metrics_to_return=[
                enums.MetricType.AP,
                enums.MetricType.AR,
                enums.MetricType.mAP,
                enums.MetricType.APAveragedOverIOUs,
                enums.MetricType.mAR,
                enums.MetricType.mAPAveragedOverIOUs,
                enums.MetricType.PrecisionRecallCurve,
            ],
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
        {"iou": 0.5, "value": 0.859, "label_key": "class"},
        {"iou": 0.75, "value": 0.761, "label_key": "class"},
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
        {"ious": iou_thresholds, "value": 0.637, "label_key": "class"}
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
        {"ious": iou_thresholds, "value": 0.652, "label_key": "class"},
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
    ), expected_value in pr_expected_answers.items():
        assert pr_metrics["value"][value][threshold][metric] == expected_value

    # now add PrecisionRecallCurve
    metrics = _compute_detection_metrics(
        db=db,
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
            convert_annotations_to_type=enums.AnnotationType.BOX,
            iou_thresholds_to_compute=list(iou_thresholds),
            iou_thresholds_to_return=[0.5, 0.75],
            metrics_to_return=[
                enums.MetricType.AP,
                enums.MetricType.AR,
                enums.MetricType.mAP,
                enums.MetricType.APAveragedOverIOUs,
                enums.MetricType.mAR,
                enums.MetricType.mAPAveragedOverIOUs,
                enums.MetricType.PrecisionRecallCurve,
            ],
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
        {"iou": 0.5, "value": 0.859, "label_key": "class"},
        {"iou": 0.75, "value": 0.761, "label_key": "class"},
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
        {"ious": iou_thresholds, "value": 0.637, "label_key": "class"}
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
        {"ious": iou_thresholds, "value": 0.652, "label_key": "class"},
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
    ), expected_value in pr_expected_answers.items():
        assert pr_metrics["value"][value][threshold][metric] == expected_value

    # finally, test the DetailedPrecisionRecallCurve version
    metrics = _compute_detection_metrics_with_detailed_precision_recall_curve(
        db=db,
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
            convert_annotations_to_type=enums.AnnotationType.BOX,
            iou_thresholds_to_compute=list(iou_thresholds),
            iou_thresholds_to_return=[0.5, 0.75],
            metrics_to_return=[
                enums.MetricType.AP,
                enums.MetricType.AR,
                enums.MetricType.mAP,
                enums.MetricType.APAveragedOverIOUs,
                enums.MetricType.mAR,
                enums.MetricType.mAPAveragedOverIOUs,
                enums.MetricType.PrecisionRecallCurve,
            ],
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
        {"iou": 0.5, "value": 0.859, "label_key": "class"},
        {"iou": 0.75, "value": 0.761, "label_key": "class"},
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
        {"ious": iou_thresholds, "value": 0.637, "label_key": "class"}
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
        {"ious": iou_thresholds, "value": 0.652, "label_key": "class"},
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

    pr_metrics = metrics[-2].model_dump(exclude_none=True)

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
    ), expected_value in pr_expected_answers.items():
        assert pr_metrics["value"][value][threshold][metric] == expected_value


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
            metrics_to_return=[
                enums.MetricType.AP,
                enums.MetricType.AR,
                enums.MetricType.mAP,
                enums.MetricType.APAveragedOverIOUs,
                enums.MetricType.mAR,
                enums.MetricType.mAPAveragedOverIOUs,
                enums.MetricType.PrecisionRecallCurve,
            ],
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
        {"iou": 0.5, "value": 0.667, "label_key": "class"},
        {"iou": 0.75, "value": 0.667, "label_key": "class"},
        # mAP METRICS AVERAGED OVER IOUS
        {"ious": iou_thresholds, "value": 0.667, "label_key": "class"},
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
        {"ious": iou_thresholds, "value": 0.667, "label_key": "class"},
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
    ), expected_value in pr_expected_answers.items():
        assert pr_metrics["value"][value][threshold][metric] == expected_value

    # test DetailedPrecisionRecallCurve version
    metrics = _compute_detection_metrics_with_detailed_precision_recall_curve(
        db=db,
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
            convert_annotations_to_type=enums.AnnotationType.RASTER,
            iou_thresholds_to_compute=list(iou_thresholds),
            iou_thresholds_to_return=[0.5, 0.75],
            metrics_to_return=[
                enums.MetricType.AP,
                enums.MetricType.AR,
                enums.MetricType.mAP,
                enums.MetricType.APAveragedOverIOUs,
                enums.MetricType.mAR,
                enums.MetricType.mAPAveragedOverIOUs,
                enums.MetricType.PrecisionRecallCurve,
            ],
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
        {"iou": 0.5, "value": 0.667, "label_key": "class"},
        {"iou": 0.75, "value": 0.667, "label_key": "class"},
        # mAP METRICS AVERAGED OVER IOUS
        {"ious": iou_thresholds, "value": 0.667, "label_key": "class"},
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
        {"ious": iou_thresholds, "value": 0.667, "label_key": "class"},
    ]

    non_pr_metrics = metrics[:-2]
    pr_metrics = metrics[-2]
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
    ), expected_value in pr_expected_answers.items():
        assert pr_metrics["value"][value][threshold][metric] == expected_value


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
        filters={"dataset_names": [dataset_name]},
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

    crud.create_groundtruths(
        db=db,
        groundtruths=[
            schemas.GroundTruth(
                dataset_name=dataset_name,
                datum=schemas.Datum(uid="uid"),
                annotations=[
                    schemas.Annotation(
                        labels=[schemas.Label(key="k1", value="v1")],
                        bounding_box=schemas.Box.from_extrema(
                            xmin=0, xmax=1, ymin=0, ymax=1
                        ),
                        is_instance=True,
                    )
                ],
            )
        ],
    )

    # test that the model does not meet the filter requirements
    # - this is b/c no predictions exist that match the evaluation task type.
    with pytest.raises(RuntimeError) as e:
        compute_detection_metrics(db=db, evaluation_id=evaluation_id)
    assert f"Model '{model_name}' does not meet filter requirements." in str(e)

    crud.create_predictions(
        db=db,
        predictions=[
            schemas.Prediction(
                dataset_name=dataset_name,
                model_name=model_name,
                datum=schemas.Datum(uid="uid"),
                annotations=[
                    schemas.Annotation(
                        labels=[
                            schemas.Label(key="k1", value="v1", score=1.0)
                        ],
                        bounding_box=schemas.Box.from_extrema(
                            xmin=0, xmax=1, ymin=0, ymax=1
                        ),
                        is_instance=True,
                    )
                ],
            )
        ],
    )

    # show that no errors raised
    compute_detection_metrics(db=db, evaluation_id=evaluation_id)
