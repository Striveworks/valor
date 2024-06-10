""" These integration tests should be run with a back end at http://localhost:8000
that is no auth
"""

import pytest

from valor import Client, Dataset, Datum, GroundTruth, Model, Prediction
from valor.enums import EvaluationStatus
from valor.schemas import Constraint, Polygon


def test_set_and_get_geospatial(
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_dets1: list[GroundTruth],
):
    coordinates = [
        [
            (125.2750725, 38.760525),
            (125.3902365, 38.775069),
            (125.5054005, 38.789613),
            (125.5051935, 38.71402425),
            (125.5049865, 38.6384355),
            (125.3902005, 38.6244225),
            (125.2754145, 38.6104095),
            (125.2752435, 38.68546725),
            (125.2750725, 38.760525),
        ]
    ]

    dataset = Dataset.create(
        name=dataset_name,
        metadata={"geospatial": Polygon(coordinates)},
    )

    # check Dataset's geospatial coordinates
    fetched_datasets = client.get_datasets()
    assert (
        fetched_datasets[0].metadata["geospatial"].get_value() == coordinates
    )

    # check Model's geospatial coordinates
    Model.create(
        name=model_name,
        metadata={"geospatial": Polygon(coordinates)},
    )

    fetched_models = client.get_models()
    assert fetched_models[0].metadata["geospatial"].get_value() == coordinates

    # check Datums's geospatial coordinates
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    expected_coords = [
        gt.datum.metadata["geospatial"].get_value() for gt in gt_dets1
    ]

    returned_datum1 = (
        dataset.get_datums()[0].metadata["geospatial"].get_value()
    )
    returned_datum2 = (
        dataset.get_datums()[1].metadata["geospatial"].get_value()
    )

    # newer datums are returned near the top
    assert expected_coords[1] == returned_datum1
    assert expected_coords[0] == returned_datum2

    dets1 = dataset.get_groundtruth("uid1")
    assert dets1
    assert dets1.datum.metadata["geospatial"].get_value() == expected_coords[0]


def test_geospatial_filter(
    client: Client,
    dataset_name,
    model_name: str,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
):
    coordinates = [
        [
            (125.2750725, 38.760525),
            (125.3902365, 38.775069),
            (125.5054005, 38.789613),
            (125.5051935, 38.71402425),
            (125.5049865, 38.6384355),
            (125.3902005, 38.6244225),
            (125.2754145, 38.6104095),
            (125.2752435, 38.68546725),
            (125.2750725, 38.760525),
        ]
    ]
    geodict = {
        "type": "Polygon",
        "coordinates": [
            [list(point) for point in subpoly] for subpoly in coordinates
        ],
    }

    dataset = Dataset.create(
        name=dataset_name, metadata={"geospatial": Polygon(coordinates)}
    )
    for gt in gt_dets1:
        gt.datum.metadata["geospatial"] = Polygon(coordinates)
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(
        name=model_name, metadata={"geospatial": Polygon(coordinates)}
    )
    for pd in pred_dets:
        pd.datum.metadata["geospatial"] = Polygon(coordinates)
        model.add_prediction(dataset, pd)
    model.finalize_inferences(dataset)

    # filtering by concatenation of datasets geospatially
    eval_job = model.evaluate_detection(
        datasets=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by={
            "dataset_metadata": {
                "geospatial": [
                    {
                        "operator": "intersect",
                        "value": geodict,
                    }
                ],
            }
        },
    )
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE
    assert len(eval_job.metrics) == 16

    # passing in an incorrectly-formatted geojson dict should return a ValueError
    geospatial_metadatum = Datum.metadata["geospatial"]
    with pytest.raises(NotImplementedError):
        model.evaluate_detection(
            dataset,
            iou_thresholds_to_compute=[0.1, 0.6],
            iou_thresholds_to_return=[0.1, 0.6],
            filter_by=[geospatial_metadatum.inside({1234: {}})],
        )

    # test datums
    eval_job = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by=[geospatial_metadatum.intersects(Polygon(coordinates))],
    )
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    assert eval_job.datum_filter.datum_metadata
    assert eval_job.datum_filter.datum_metadata["geospatial"] == [
        Constraint(value=geodict, operator="intersect")
    ]
    assert len(eval_job.metrics) == 16

    # filtering by model is allowed, this is the equivalent of requesting..
    # "Give me the dataset that model A has operated over."
    eval_job = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by={
            "model_metadata": {
                "geospatial": [
                    {
                        "operator": "inside",
                        "value": {
                            "type": "polygon",
                            "coordinates": [
                                [
                                    [124.0, 37.0],
                                    [128.0, 37.0],
                                    [128.0, 40.0],
                                    [124.0, 40.0],
                                ]
                            ],
                        },
                    }
                ],
            }
        },
    )
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE
