""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""
from dataclasses import asdict

from velour import Dataset, GroundTruth, Model, Prediction
from velour.client import Client


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
        client=client, name=dataset_name, geospatial=geo_dict
    )

    # check Dataset's geospatial coordinates
    fetched_datasets = client.get_datasets()
    assert fetched_datasets[0]["geospatial"] == geo_dict

    # check Model's geospatial coordinates
    Model.create(client=client, name=model_name, geospatial=geo_dict)
    fetched_models = client.get_models()
    assert fetched_models[0]["geospatial"] == geo_dict

    # check Datums's geospatial coordinates
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    expected_coords = [gt.datum.geospatial for gt in gt_dets1]

    returned_datum1 = dataset.get_datums()[0].geospatial
    returned_datum2 = dataset.get_datums()[1].geospatial

    assert expected_coords[0] == returned_datum1
    assert expected_coords[1] == returned_datum2

    dets1 = dataset.get_groundtruth("uid1")

    assert dets1.datum.geospatial == expected_coords[0]


def test_geospatial_filter(
    client: Client,
    dataset_name: str,
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
        client=client, name=dataset_name, geospatial=geo_dict
    )
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(client=client, name=model_name, geospatial=geo_dict)
    for pd in pred_dets:
        model.add_prediction(pd)
    model.finalize_inferences(dataset)

    # test filtering for the dataset
    eval_job = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters={
            "dataset_geospatial": [
                {
                    "operator": "outside",
                    "value": {
                        "geometry": {"type": "Point", "coordinates": [0, 0]}
                    },
                }
            ],
        },
        timeout=30,
    )

    settings = asdict(eval_job.settings)
    assert settings["settings"]["filters"]["dataset_geospatial"] == [
        {
            "value": {
                "geometry": {
                    "type": "Point",
                    "coordinates": [0.0, 0.0],
                }
            },
            "operator": "outside",
        }
    ]

    assert len(eval_job.metrics["metrics"]) > 0

    # test dataset WHERE miss
    eval_job = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters={
            "dataset_geospatial": [
                {
                    "operator": "inside",
                    "value": {
                        "geometry": {"type": "Point", "coordinates": [0, 0]}
                    },
                }
            ],
        },
        timeout=30,
    )

    settings = asdict(eval_job.settings)
    assert settings["settings"]["filters"]["dataset_geospatial"] == [
        {
            "value": {
                "geometry": {
                    "type": "Point",
                    "coordinates": [0.0, 0.0],
                }
            },
            "operator": "inside",
        }
    ]

    assert len(eval_job.metrics["metrics"]) == 0

    # test datums
    eval_job = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters={
            "datum_geospatial": [
                {
                    "operator": "inside",
                    "value": {
                        "geometry": {"type": "Point", "coordinates": [0, 0]}
                    },
                }
            ],
        },
        timeout=30,
    )

    settings = asdict(eval_job.settings)
    assert settings["settings"]["filters"]["datum_geospatial"] == [
        {
            "value": {
                "geometry": {
                    "type": "Point",
                    "coordinates": [0.0, 0.0],
                }
            },
            "operator": "inside",
        }
    ]

    assert len(eval_job.metrics["metrics"]) == 0

    # test models
    eval_job = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters={
            "models_geospatial": [
                {
                    "operator": "inside",
                    "value": {
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [124.0, 37.0],
                                    [128.0, 37.0],
                                    [128.0, 40.0],
                                    [124.0, 40.0],
                                ]
                            ],
                        }
                    },
                }
            ],
        },
        timeout=30,
    )

    settings = asdict(eval_job.settings)
    assert settings["settings"]["filters"]["models_geospatial"] == [
        {
            "value": {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [124.0, 37.0],
                            [128.0, 37.0],
                            [128.0, 40.0],
                            [124.0, 40.0],
                        ]
                    ],
                }
            },
            "operator": "inside",
        }
    ]

    assert len(eval_job.metrics["metrics"]) > 0
