""" These integration tests should be run with a back end at http://localhost:8000
that is no auth
"""

import pytest

from valor import (
    Client,
    Constraint,
    Dataset,
    Datum,
    GroundTruth,
    Model,
    Prediction,
)
from valor.enums import EvaluationStatus


def test_set_and_get_geospatial(
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_dets1: list[GroundTruth],
):
    coordinates = [
        [
            [125.2750725, 38.760525],
            [125.3902365, 38.775069],
            [125.5054005, 38.789613],
            [125.5051935, 38.71402425],
            [125.5049865, 38.6384355],
            [125.3902005, 38.6244225],
            [125.2754145, 38.6104095],
            [125.2752435, 38.68546725],
            [125.2750725, 38.760525],
        ]
    ]
    geo_dict = {"type": "Polygon", "coordinates": coordinates}

    dataset = Dataset.create(
        name=dataset_name,
        metadata={"geospatial": geo_dict},
    )

    # check Dataset's geospatial coordinates
    fetched_datasets = client.get_datasets()
    assert fetched_datasets[0].metadata["geospatial"] == geo_dict

    # check Model's geospatial coordinates
    Model.create(
        name=model_name,
        metadata={"geospatial": geo_dict},
    )

    fetched_models = client.get_models()
    assert fetched_models[0].metadata["geospatial"] == geo_dict

    # check Datums's geospatial coordinates
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    expected_coords = [gt.datum.metadata["geospatial"] for gt in gt_dets1]

    returned_datum1 = dataset.get_datums()[0].metadata["geospatial"]
    returned_datum2 = dataset.get_datums()[1].metadata["geospatial"]

    assert expected_coords[0] == returned_datum1
    assert expected_coords[1] == returned_datum2

    dets1 = dataset.get_groundtruth("uid1")
    assert dets1
    assert dets1.datum.metadata["geospatial"] == expected_coords[0]


def test_geospatial_filter(
    client: Client,
    dataset_name,
    model_name: str,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
):
    coordinates = [
        [
            [125.2750725, 38.760525],
            [125.3902365, 38.775069],
            [125.5054005, 38.789613],
            [125.5051935, 38.71402425],
            [125.5049865, 38.6384355],
            [125.3902005, 38.6244225],
            [125.2754145, 38.6104095],
            [125.2752435, 38.68546725],
            [125.2750725, 38.760525],
        ]
    ]
    geo_dict = {"type": "Polygon", "coordinates": coordinates}

    dataset = Dataset.create(
        name=dataset_name, metadata={"geospatial": geo_dict}
    )
    for gt in gt_dets1:
        gt.datum.metadata["geospatial"] = geo_dict
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(name=model_name, metadata={"geospatial": geo_dict})
    for pd in pred_dets:
        gt.datum.metadata["geospatial"] = geo_dict  # type: ignore - __setitem__ possibly unbound; shouldn't matter in this case
        model.add_prediction(dataset, pd)
    model.finalize_inferences(dataset)

    # filtering by concatenation of datasets geospatially
    eval_job = model.evaluate_detection(
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by={
            "dataset_metadata": {
                "geospatial": [
                    {
                        "operator": "intersect",
                        "value": geo_dict,
                    }
                ],
            }
        },
    )
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE
    assert len(eval_job.metrics) == 12

    # passing in an incorrectly-formatted geojson dict should return a ValueError
    geospatial_metadata = Datum.metadata["geospatial"]
    with pytest.raises(NotImplementedError) as e:
        model.evaluate_detection(
            dataset,
            iou_thresholds_to_compute=[0.1, 0.6],
            iou_thresholds_to_return=[0.1, 0.6],
            filter_by=[
                geospatial_metadata.inside({"incorrectly_formatted_dict": {}})  # type: ignore - filter type error
            ],
        )
    assert "is not supported" in str(e)
    # test datums
    eval_job = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by=[geospatial_metadata.intersect(geo_dict)],  # type: ignore - filter type error
    )
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    assert eval_job.datum_filter.datum_metadata
    assert eval_job.datum_filter.datum_metadata["geospatial"] == [
        Constraint(value=geo_dict, operator="intersect")
    ]
    assert len(eval_job.metrics) == 12

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
                            "type": "Polygon",
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
