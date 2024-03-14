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
        -6056857970499991270: [
            RankedPair(
                gt_id=50, pd_id=48, score=0.953, iou=0.8775260257195348
            ),
            RankedPair(
                gt_id=49, pd_id=47, score=0.805, iou=0.8811645870469409
            ),
            RankedPair(gt_id=47, pd_id=45, score=0.611, iou=0.742765273311898),
            RankedPair(
                gt_id=46, pd_id=44, score=0.407, iou=0.8970133882595271
            ),
            RankedPair(
                gt_id=48, pd_id=46, score=0.335, iou=1.0000000000000002
            ),
        ],
        -5475278634838695825: [
            RankedPair(
                gt_id=55, pd_id=53, score=0.883, iou=0.9999999999999992
            ),
            RankedPair(
                gt_id=53, pd_id=51, score=0.782, iou=0.8911860718171924
            ),
            RankedPair(
                gt_id=58, pd_id=55, score=0.561, iou=0.8809523809523806
            ),
            RankedPair(
                gt_id=51, pd_id=49, score=0.532, iou=0.9999999999999998
            ),
            RankedPair(
                gt_id=60, pd_id=57, score=0.349, iou=0.6093750000000003
            ),
            RankedPair(
                gt_id=57, pd_id=54, score=0.271, iou=0.8562185478073326
            ),
            RankedPair(
                gt_id=59, pd_id=50, score=0.204, iou=0.8089209038203885
            ),
            RankedPair(
                gt_id=52, pd_id=56, score=0.204, iou=0.3460676561905953
            ),
            RankedPair(
                gt_id=54, pd_id=52, score=0.202, iou=0.6719967199671995
            ),
        ],
        7745592076607202088: [
            RankedPair(gt_id=43, pd_id=41, score=0.726, iou=0.9213161659513592)
        ],
        -7103043785393643334: [
            RankedPair(
                gt_id=44, pd_id=42, score=0.546, iou=0.8387196824018363
            ),
            RankedPair(
                gt_id=41, pd_id=39, score=0.236, iou=0.7756590016825575
            ),
        ],
        -8471527583113776199: [
            RankedPair(gt_id=45, pd_id=43, score=0.3, iou=0.8596978106691334)
        ],
    }

    grouper_mappings = {
        "label_id_to_grouper_id_mapping": {
            16: -6056857970499991270,
            18: -136394889496151664,
            13: -7103043785393643334,
            15: -8471527583113776199,
            14: 7745592076607202088,
            17: -5475278634838695825,
        },
        "grouper_id_to_label_ids_mapping": {
            -6056857970499991270: [16],
            -136394889496151664: [18],
            -7103043785393643334: [13],
            -8471527583113776199: [15],
            7745592076607202088: [14],
            -5475278634838695825: [17],
        },
        "grouper_id_to_grouper_label_mapping": {
            -6056857970499991270: schemas.Label(
                key="class", value="0", score=None
            ),
            -136394889496151664: schemas.Label(
                key="class", value="3", score=None
            ),
            -7103043785393643334: schemas.Label(
                key="class", value="4", score=None
            ),
            -8471527583113776199: schemas.Label(
                key="class", value="1", score=None
            ),
            7745592076607202088: schemas.Label(
                key="class", value="2", score=None
            ),
            -5475278634838695825: schemas.Label(
                key="class", value="49", score=None
            ),
        },
    }

    number_of_groundtruths_per_grouper = {
        -6056857970499991270: 5,
        -5475278634838695825: 10,
        7745592076607202088: 2,
        -7103043785393643334: 2,
        -8471527583113776199: 1,
    }

    mismatched_entries = [(None, -136394889496151664, 0.318)]

    output = _compute_curves(
        sorted_ranked_pairs=sorted_ranked_pairs,
        grouper_mappings=grouper_mappings,
        number_of_groundtruths_per_grouper=number_of_groundtruths_per_grouper,
        mismatched_entries=mismatched_entries,
        iou_threshold=0.5,
    )

    expected_values = {
        schemas.Label(key="class", value="0", score=None): {
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
        schemas.Label(key="class", value="3", score=None): {
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
        schemas.Label(key="class", value="4", score=None): {
            0.05: {
                "tp": 2,  # two tps with scores [.236, .546]
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
        schemas.Label(key="class", value="1", score=None): {
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
        schemas.Label(key="class", value="2", score=None): {
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
        schemas.Label(key="class", value="49", score=None): {
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
    }

    assert output == expected_values


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
