from sqlalchemy.orm import Session

from valor_api import enums, schemas
from valor_api.backend.metrics.detection import (
    RankedPair,
    _compute_curves,
    _compute_detection_metrics,
)
from valor_api.backend.models import GroundTruth, Prediction


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
        1233776554513384492: [
            RankedPair(
                gt_id=392, pd_id=327, score=0.953, iou=0.8775260257195348
            ),
            RankedPair(
                gt_id=391, pd_id=326, score=0.805, iou=0.8811645870469409
            ),
            RankedPair(
                gt_id=389, pd_id=324, score=0.611, iou=0.742765273311898
            ),
            RankedPair(
                gt_id=388, pd_id=323, score=0.407, iou=0.8970133882595271
            ),
            RankedPair(
                gt_id=390, pd_id=325, score=0.335, iou=1.0000000000000002
            ),
        ],
        -5190670243438198747: [
            RankedPair(
                gt_id=397, pd_id=332, score=0.883, iou=0.9999999999999992
            ),
            RankedPair(
                gt_id=395, pd_id=330, score=0.782, iou=0.8911860718171924
            ),
            RankedPair(
                gt_id=400, pd_id=334, score=0.561, iou=0.8809523809523806
            ),
            RankedPair(
                gt_id=393, pd_id=328, score=0.532, iou=0.9999999999999998
            ),
            RankedPair(
                gt_id=402, pd_id=336, score=0.349, iou=0.6093750000000003
            ),
            RankedPair(
                gt_id=399, pd_id=333, score=0.271, iou=0.8562185478073326
            ),
            RankedPair(
                gt_id=401, pd_id=329, score=0.204, iou=0.8089209038203885
            ),
            RankedPair(
                gt_id=394, pd_id=335, score=0.204, iou=0.3460676561905953
            ),
            RankedPair(
                gt_id=396, pd_id=331, score=0.202, iou=0.6719967199671995
            ),
        ],
        3723564822429385001: [
            RankedPair(
                gt_id=385, pd_id=320, score=0.726, iou=0.9213161659513592
            )
        ],
        -8234252136290100573: [
            RankedPair(
                gt_id=386, pd_id=321, score=0.546, iou=0.8387196824018363
            ),
            RankedPair(
                gt_id=383, pd_id=318, score=0.236, iou=0.7756590016825575
            ),
        ],
        -62387667849561155: [
            RankedPair(gt_id=387, pd_id=322, score=0.3, iou=0.8596978106691334)
        ],
    }

    grouper_mappings = {
        "label_id_to_grouper_id_mapping": {
            260: -8234252136290100573,
            261: 3723564822429385001,
            265: -1317487234205701345,
            262: 1233776554513384492,
            264: -5190670243438198747,
            263: -62387667849561155,
        },
        "grouper_id_to_label_ids_mapping": {
            -8234252136290100573: [260],
            3723564822429385001: [261],
            -1317487234205701345: [265],
            1233776554513384492: [262],
            -5190670243438198747: [264],
            -62387667849561155: [263],
        },
        "grouper_id_to_grouper_label_mapping": {
            -8234252136290100573: schemas.Label(
                key="class", value="4", score=None
            ),
            3723564822429385001: schemas.Label(
                key="class", value="2", score=None
            ),
            -1317487234205701345: schemas.Label(
                key="class", value="3", score=None
            ),
            1233776554513384492: schemas.Label(
                key="class", value="0", score=None
            ),
            -5190670243438198747: schemas.Label(
                key="class", value="49", score=None
            ),
            -62387667849561155: schemas.Label(
                key="class", value="1", score=None
            ),
        },
    }

    number_of_groundtruths_per_grouper = {
        -8234252136290100573: 2,
        3723564822429385001: 2,
        -62387667849561155: 1,
        1233776554513384492: 5,
        -5190670243438198747: 10,
    }

    false_positive_entries = [(None, -1317487234205701345, 0.318)]

    output = _compute_curves(
        sorted_ranked_pairs=sorted_ranked_pairs,
        grouper_mappings=grouper_mappings,
        number_of_groundtruths_per_grouper=number_of_groundtruths_per_grouper,
        false_positive_entries=false_positive_entries,
        iou_threshold=0.5,
    )

    expected_values = [
        schemas.PrecisionRecallCurve(
            label_key="class",
            value={
                "4": {
                    0.05: {
                        "tp": 2,
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1_score": 1.0,
                    },
                    0.1: {
                        "tp": 2,
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1_score": 1.0,
                    },
                    0.15: {
                        "tp": 2,
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1_score": 1.0,
                    },
                    0.2: {
                        "tp": 2,
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1_score": 1.0,
                    },
                    0.25: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.3: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.35: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.4: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.45: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.5: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.55: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 2,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.6: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 2,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.65: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 2,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.7: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 2,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.75: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 2,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.8: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 2,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.85: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 2,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.9: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 2,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.95: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 2,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                },
                "2": {
                    0.05: {
                        "tp": 1,  # one correct classification with score .726
                        "fp": 0,
                        "fn": 1,  # one incorrect classification
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.1: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.15: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.2: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.25: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.3: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.35: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.4: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.45: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.5: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.55: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.6: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.65: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.7: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.75: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 2,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.8: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 2,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.85: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 2,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.9: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 2,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.95: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 2,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                },
                "3": {
                    0.05: {
                        "tp": 0,
                        "fp": 1,  # there are no gts with this label, only one preidctionw with a score of .318
                        "fn": 0,
                        "precision": 0.0,
                        "recall": -1,
                        "f1_score": -1,
                    },
                    0.1: {
                        "tp": 0,
                        "fp": 1,
                        "fn": 0,
                        "precision": 0.0,
                        "recall": -1,
                        "f1_score": -1,
                    },
                    0.15: {
                        "tp": 0,
                        "fp": 1,
                        "fn": 0,
                        "precision": 0.0,
                        "recall": -1,
                        "f1_score": -1,
                    },
                    0.2: {
                        "tp": 0,
                        "fp": 1,
                        "fn": 0,
                        "precision": 0.0,
                        "recall": -1,
                        "f1_score": -1,
                    },
                    0.25: {
                        "tp": 0,
                        "fp": 1,
                        "fn": 0,
                        "precision": 0.0,
                        "recall": -1,
                        "f1_score": -1,
                    },
                    0.3: {
                        "tp": 0,
                        "fp": 1,
                        "fn": 0,
                        "precision": 0.0,
                        "recall": -1,
                        "f1_score": -1,
                    },
                    0.35: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "precision": -1,
                        "recall": -1,
                        "f1_score": -1.0,
                    },
                    0.4: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "precision": -1,
                        "recall": -1,
                        "f1_score": -1.0,
                    },
                    0.45: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "precision": -1,
                        "recall": -1,
                        "f1_score": -1.0,
                    },
                    0.5: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "precision": -1,
                        "recall": -1,
                        "f1_score": -1.0,
                    },
                    0.55: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "precision": -1,
                        "recall": -1,
                        "f1_score": -1.0,
                    },
                    0.6: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "precision": -1,
                        "recall": -1,
                        "f1_score": -1.0,
                    },
                    0.65: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "precision": -1,
                        "recall": -1,
                        "f1_score": -1.0,
                    },
                    0.7: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "precision": -1,
                        "recall": -1,
                        "f1_score": -1.0,
                    },
                    0.75: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "precision": -1,
                        "recall": -1,
                        "f1_score": -1.0,
                    },
                    0.8: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "precision": -1,
                        "recall": -1,
                        "f1_score": -1.0,
                    },
                    0.85: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "precision": -1,
                        "recall": -1,
                        "f1_score": -1.0,
                    },
                    0.9: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "precision": -1,
                        "recall": -1,
                        "f1_score": -1.0,
                    },
                    0.95: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "precision": -1,
                        "recall": -1,
                        "f1_score": -1.0,
                    },
                },
                "0": {
                    0.05: {
                        "tp": 5,  # start with five true positives, then these predictions convert to fn using scores [0.407, 0.611, 0.335, 0.805, 0.953]
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1_score": 1.0,
                    },
                    0.1: {
                        "tp": 5,
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1_score": 1.0,
                    },
                    0.15: {
                        "tp": 5,
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1_score": 1.0,
                    },
                    0.2: {
                        "tp": 5,
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1_score": 1.0,
                    },
                    0.25: {
                        "tp": 5,
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1_score": 1.0,
                    },
                    0.3: {
                        "tp": 5,
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1_score": 1.0,
                    },
                    0.35: {
                        "tp": 4,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.8,
                        "f1_score": 0.888888888888889,
                    },
                    0.4: {
                        "tp": 4,
                        "fp": 0,
                        "fn": 1,
                        "precision": 1.0,
                        "recall": 0.8,
                        "f1_score": 0.888888888888889,
                    },
                    0.45: {
                        "tp": 3,
                        "fp": 0,
                        "fn": 2,
                        "precision": 1.0,
                        "recall": 0.6,
                        "f1_score": 0.7499999999999999,
                    },
                    0.5: {
                        "tp": 3,
                        "fp": 0,
                        "fn": 2,
                        "precision": 1.0,
                        "recall": 0.6,
                        "f1_score": 0.7499999999999999,
                    },
                    0.55: {
                        "tp": 3,
                        "fp": 0,
                        "fn": 2,
                        "precision": 1.0,
                        "recall": 0.6,
                        "f1_score": 0.7499999999999999,
                    },
                    0.6: {
                        "tp": 3,
                        "fp": 0,
                        "fn": 2,
                        "precision": 1.0,
                        "recall": 0.6,
                        "f1_score": 0.7499999999999999,
                    },
                    0.65: {
                        "tp": 2,
                        "fp": 0,
                        "fn": 3,
                        "precision": 1.0,
                        "recall": 0.4,
                        "f1_score": 0.5714285714285715,
                    },
                    0.7: {
                        "tp": 2,
                        "fp": 0,
                        "fn": 3,
                        "precision": 1.0,
                        "recall": 0.4,
                        "f1_score": 0.5714285714285715,
                    },
                    0.75: {
                        "tp": 2,
                        "fp": 0,
                        "fn": 3,
                        "precision": 1.0,
                        "recall": 0.4,
                        "f1_score": 0.5714285714285715,
                    },
                    0.8: {
                        "tp": 2,
                        "fp": 0,
                        "fn": 3,
                        "precision": 1.0,
                        "recall": 0.4,
                        "f1_score": 0.5714285714285715,
                    },
                    0.85: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 4,
                        "precision": 1.0,
                        "recall": 0.2,
                        "f1_score": 0.33333333333333337,
                    },
                    0.9: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 4,
                        "precision": 1.0,
                        "recall": 0.2,
                        "f1_score": 0.33333333333333337,
                    },
                    0.95: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 4,
                        "precision": 1.0,
                        "recall": 0.2,
                        "f1_score": 0.33333333333333337,
                    },
                },
                "49": {
                    0.05: {
                        "tp": 8,  # there are 10 gts and 9 preds with this label, but one has an IOU of .3
                        "fp": 0,
                        "fn": 2,
                        "precision": 1.0,
                        "recall": 0.8,
                        "f1_score": 0.888888888888889,
                    },
                    0.1: {
                        "tp": 8,
                        "fp": 0,
                        "fn": 2,
                        "precision": 1.0,
                        "recall": 0.8,
                        "f1_score": 0.888888888888889,
                    },
                    0.15: {
                        "tp": 8,
                        "fp": 0,
                        "fn": 2,
                        "precision": 1.0,
                        "recall": 0.8,
                        "f1_score": 0.888888888888889,
                    },
                    0.2: {
                        "tp": 8,
                        "fp": 0,
                        "fn": 2,
                        "precision": 1.0,
                        "recall": 0.8,
                        "f1_score": 0.888888888888889,
                    },
                    0.25: {
                        "tp": 6,
                        "fp": 0,
                        "fn": 4,
                        "precision": 1.0,
                        "recall": 0.6,
                        "f1_score": 0.7499999999999999,
                    },
                    0.3: {
                        "tp": 5,
                        "fp": 0,
                        "fn": 5,
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                    },
                    0.35: {
                        "tp": 4,
                        "fp": 0,
                        "fn": 6,
                        "precision": 1.0,
                        "recall": 0.4,
                        "f1_score": 0.5714285714285715,
                    },
                    0.4: {
                        "tp": 4,
                        "fp": 0,
                        "fn": 6,
                        "precision": 1.0,
                        "recall": 0.4,
                        "f1_score": 0.5714285714285715,
                    },
                    0.45: {
                        "tp": 4,
                        "fp": 0,
                        "fn": 6,
                        "precision": 1.0,
                        "recall": 0.4,
                        "f1_score": 0.5714285714285715,
                    },
                    0.5: {
                        "tp": 4,
                        "fp": 0,
                        "fn": 6,
                        "precision": 1.0,
                        "recall": 0.4,
                        "f1_score": 0.5714285714285715,
                    },
                    0.55: {
                        "tp": 3,
                        "fp": 0,
                        "fn": 7,
                        "precision": 1.0,
                        "recall": 0.3,
                        "f1_score": 0.4615384615384615,
                    },
                    0.6: {
                        "tp": 2,
                        "fp": 0,
                        "fn": 8,
                        "precision": 1.0,
                        "recall": 0.2,
                        "f1_score": 0.33333333333333337,
                    },
                    0.65: {
                        "tp": 2,
                        "fp": 0,
                        "fn": 8,
                        "precision": 1.0,
                        "recall": 0.2,
                        "f1_score": 0.33333333333333337,
                    },
                    0.7: {
                        "tp": 2,
                        "fp": 0,
                        "fn": 8,
                        "precision": 1.0,
                        "recall": 0.2,
                        "f1_score": 0.33333333333333337,
                    },
                    0.75: {
                        "tp": 2,
                        "fp": 0,
                        "fn": 8,
                        "precision": 1.0,
                        "recall": 0.2,
                        "f1_score": 0.33333333333333337,
                    },
                    0.8: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 9,
                        "precision": 1.0,
                        "recall": 0.1,
                        "f1_score": 0.18181818181818182,
                    },
                    0.85: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 9,
                        "precision": 1.0,
                        "recall": 0.1,
                        "f1_score": 0.18181818181818182,
                    },
                    0.9: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 10,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.95: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 10,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                },
                "1": {
                    0.05: {
                        "tp": 1,  # one tp with score .3
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1_score": 1.0,
                    },
                    0.1: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1_score": 1.0,
                    },
                    0.15: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1_score": 1.0,
                    },
                    0.2: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1_score": 1.0,
                    },
                    0.25: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1_score": 1.0,
                    },
                    0.3: {
                        "tp": 1,
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1_score": 1.0,
                    },
                    0.35: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 1,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.4: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 1,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.45: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 1,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.5: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 1,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.55: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 1,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.6: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 1,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.65: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 1,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.7: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 1,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.75: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 1,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.8: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 1,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.85: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 1,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.9: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 1,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                    0.95: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 1,
                        "precision": -1,
                        "recall": 0.0,
                        "f1_score": -1,
                    },
                },
            },
            pr_curve_iou_threshold=0.5,
        )
    ]

    assert output == expected_values

    # do a second test with a much higher iou_threshold
    second_output = _compute_curves(
        sorted_ranked_pairs=sorted_ranked_pairs,
        grouper_mappings=grouper_mappings,
        number_of_groundtruths_per_grouper=number_of_groundtruths_per_grouper,
        false_positive_entries=false_positive_entries,
        iou_threshold=0.9,
    )

    assert second_output[0].pr_curve_iou_threshold == 0.9
    assert (
        second_output[0].value["4"][0.05]["fn"]
        == 2  # both predictions have an iou below .9
    )
    assert (
        second_output[0].value["2"][0.05]["tp"]
        == 1  # not affected since over iou threshold of .9
    )
    assert (
        second_output[0].value["3"][0.05]["fp"] == 1
    )  # not affected since iou_threshold doesn't matter without a ground truth to compare against
    assert (
        second_output[0].value["0"][0.05]["tp"]
        == 1  # all predictions except for one are disqualified by iou_threshold
    )
    assert (
        second_output[0].value["49"][0.05]["tp"]
        == 2  # 7 predictions are disqualified due to iou_threshold, one was already disqualified in the first test
    )
    assert (
        second_output[0].value["1"][0.05]["tp"]
        == 0  # 1 prediction disqualified
    )


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
            iou_thresholds_to_compute=iou_thresholds,
            iou_thresholds_to_return=[0.5, 0.75],
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

    metrics = [m.model_dump(exclude_none=True) for m in metrics]

    for m in metrics:
        _round_dict(m, 3)

    # cf with torch metrics/pycocotools results listed here:
    # https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    expected = [
        # AP METRICS
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
        # mAP METRICS
        {"iou": 0.5, "value": 0.859},
        {"iou": 0.75, "value": 0.761},
        # AP METRICS AVERAGED OVER IOUS
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
        # mAP METRICS AVERAGED OVER IOUS
        {"ious": iou_thresholds, "value": 0.637},
        # AR METRICS
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
        # mAR METRICS
        {
            "ious": iou_thresholds,
            "value": 0.652,
        },
    ]

    assert len(metrics) == len(expected)

    # sort labels lists
    for m in metrics + expected:
        if "labels" in m:
            m["labels"] = sorted(m["labels"], key=lambda x: x["value"])

    # check that metrics and labels are equivalent
    for m in metrics:
        assert m in expected

    for m in expected:
        assert m in metrics
