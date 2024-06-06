""" These integration tests should be run with a back end at http://localhost:8000
that is no auth
"""

import io
import json
import warnings
from typing import Any

import numpy as np
import PIL.Image
import pytest
from geoalchemy2.functions import ST_AsPNG
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from valor import (
    Annotation,
    Client,
    Dataset,
    Datum,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from valor.exceptions import ClientException
from valor_api.backend import models


def _list_of_points_from_wkt_polygon(
    db: Session, det: models.Annotation
) -> list[tuple[float, float]]:
    geo = json.loads(db.scalar(det.polygon.ST_AsGeoJSON()) or "")
    assert len(geo["coordinates"]) == 1
    return [(p[0], p[1]) for p in geo["coordinates"][0]]


def _test_create_model_with_preds(
    client: Client,
    dataset_name: str,
    model_name: str,
    gts: list[Any],
    preds: list[Any],
    preds_model_class: type,
    preds_expected_number: int,
    expected_labels_tuples: set[tuple[str, str]],
    expected_scores: set[float],
    db: Session,
):
    """Tests that the client can be used to add predictions.

    Parameters
    ----------
    client
    gts
        list of groundtruth objects (from `valor.data_types`)
    preds
        list of prediction objects (from `valor.data_types`)
    preds_model_class
        class in `valor_api.models` that specifies the labeled predictions
    preds_expected_number
        expected number of (labeled) predictions added to the database
    expected_labels_tuples
        set of tuples of key/value labels to check were added to the database
    expected_scores
        set of the scores of hte predictions
    db

    Returns
    -------
    the sqlalchemy objects for the created predictions
    """
    dataset = Dataset.create(dataset_name)
    model = Model.create(model_name)

    # verify we get an error if we try to create another model
    # with the same name
    with pytest.raises(ClientException) as exc_info:
        client.create_model({"name": model_name})
    assert "already exists" in str(exc_info)

    # add groundtruths
    for gt in gts:
        dataset.add_groundtruth(gt)

    # finalize dataset
    dataset.finalize()

    # add predictions
    for pd in preds:
        model.add_prediction(dataset, pd)

    # check predictions have been added
    db_preds = db.scalars(select(preds_model_class)).all()
    assert len(db_preds) == preds_expected_number

    # check labels
    assert (
        set([(p.label.key, p.label.value) for p in db_preds])
        == expected_labels_tuples
    )

    # check scores
    assert set([p.score for p in db_preds]) == expected_scores

    # check that the get_model method works
    retrieved_model = Model.get(model_name)
    assert isinstance(retrieved_model, type(model))
    assert retrieved_model.name == model_name

    return db_preds


def test_create_model_with_href_and_description(
    db: Session,
    client: Client,
    model_name: str,
):
    href = "http://a.com/b"
    description = "a description"
    Model.create(
        model_name,
        metadata={
            "href": href,
            "description": description,
        },
    )

    model_id = db.scalar(
        select(models.Model.id).where(models.Model.name == model_name)
    )
    assert isinstance(model_id, int)

    model_metadata = db.scalar(
        select(models.Model.meta).where(models.Model.name == model_name)
    )
    assert model_metadata == {
        "href": href,
        "description": description,
    }


def test_create_image_model_with_predicted_detections(
    db: Session,
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_poly_dets1: list[GroundTruth],
    pred_poly_dets: list[Prediction],
):
    labeled_pred_dets = _test_create_model_with_preds(
        client=client,
        dataset_name=dataset_name,
        model_name=model_name,
        gts=gt_poly_dets1,
        preds=pred_poly_dets,
        preds_model_class=models.Prediction,
        preds_expected_number=2,
        expected_labels_tuples={("k1", "v1"), ("k2", "v2")},
        expected_scores={0.3, 0.98},
        db=db,
    )

    # get db polygon
    db_annotation_ids = {pred.annotation_id for pred in labeled_pred_dets}
    db_annotations = [
        db.scalar(
            select(models.Annotation).where(
                and_(
                    models.Annotation.id == id,
                    models.Annotation.model_id.isnot(None),
                )
            )
        )
        for id in db_annotation_ids
    ]
    db_point_lists = [
        _list_of_points_from_wkt_polygon(db, annotation)
        for annotation in db_annotations
    ]

    # get fixture polygons
    fx_point_lists = []
    for pd in pred_poly_dets:
        for ann in pd.annotations:
            assert ann.polygon is not None
            fx_point_lists.append(ann.polygon.boundary)

    # check boundary
    for fx_points in fx_point_lists:
        assert fx_points in db_point_lists


def test_create_model_with_predicted_segmentations(
    db: Session,
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_segs: list[GroundTruth],
    pred_instance_segs: list[Prediction],
):
    """Tests that we can create a predicted segmentation from a mask array"""
    _test_create_model_with_preds(
        client=client,
        dataset_name=dataset_name,
        model_name=model_name,
        gts=gt_segs,
        preds=pred_instance_segs,
        preds_model_class=models.Prediction,
        preds_expected_number=2,
        expected_labels_tuples={("k1", "v1"), ("k2", "v2")},
        expected_scores={0.87, 0.92},
        db=db,
    )

    # grab the segmentation from the db, recover the mask, and check
    # its equal to the mask the client sent over
    db_annotations = (
        db.query(models.Annotation)
        .where(models.Annotation.model_id.isnot(None))
        .all()
    )

    if db_annotations[0].datum_id < db_annotations[1].datum_id:
        raster_uid1 = db_annotations[0].raster
        raster_uid2 = db_annotations[1].raster
    else:
        raster_uid1 = db_annotations[1].raster
        raster_uid2 = db_annotations[0].raster

    # test raster 1
    png_from_db = db.scalar(ST_AsPNG(raster_uid1))
    f = io.BytesIO(png_from_db.tobytes())
    mask_array = np.array(PIL.Image.open(f))
    assert pred_instance_segs[0].annotations[0].raster is not None
    np.testing.assert_equal(
        mask_array, pred_instance_segs[0].annotations[0].raster.array
    )

    # test raster 2
    png_from_db = db.scalar(ST_AsPNG(raster_uid2))
    f = io.BytesIO(png_from_db.tobytes())
    mask_array = np.array(PIL.Image.open(f))
    assert pred_instance_segs[1].annotations[0].raster is not None
    np.testing.assert_equal(
        mask_array, pred_instance_segs[1].annotations[0].raster.array
    )


def test_create_image_model_with_predicted_classifications(
    db: Session,
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_clfs: list[GroundTruth],
    pred_clfs: list[Prediction],
):
    _test_create_model_with_preds(
        client=client,
        dataset_name=dataset_name,
        model_name=model_name,
        gts=gt_clfs,
        preds=pred_clfs,
        preds_model_class=models.Prediction,
        preds_expected_number=6,
        expected_labels_tuples={
            ("k5", "v1"),
            ("k3", "v1"),
            ("k4", "v5"),
            ("k4", "v1"),
            ("k4", "v8"),
            ("k4", "v4"),
        },
        expected_scores={0.47, 0.53, 1.0, 0.71, 0.29},
        db=db,
    )


def test_client_delete_model(
    db: Session,
    client: Client,
    model_name: str,
):
    Model.create(model_name)
    assert db.scalar(select(func.count(models.Model.name))) == 1
    client.delete_model(model_name, timeout=30)
    assert db.scalar(select(func.count(models.Model.name))) == 0


def test_create_tabular_model_with_predicted_classifications(
    db: Session,
    client: Client,
    dataset_name: str,
    model_name: str,
):
    _test_create_model_with_preds(
        client=client,
        dataset_name=dataset_name,
        model_name=model_name,
        gts=[
            GroundTruth(
                datum=Datum(uid="uid1"),
                annotations=[
                    Annotation(
                        labels=[
                            Label(key="k1", value="v1"),
                            Label(key="k2", value="v2"),
                        ],
                    )
                ],
            ),
            GroundTruth(
                datum=Datum(uid="uid2"),
                annotations=[
                    Annotation(
                        labels=[Label(key="k1", value="v3")],
                    )
                ],
            ),
        ],
        preds=[
            Prediction(
                datum=Datum(uid="uid1"),
                annotations=[
                    Annotation(
                        labels=[
                            Label(key="k1", value="v1", score=0.6),
                            Label(key="k1", value="v2", score=0.4),
                            Label(key="k2", value="v6", score=1.0),
                        ],
                    )
                ],
            ),
            Prediction(
                datum=Datum(uid="uid2"),
                annotations=[
                    Annotation(
                        labels=[
                            Label(key="k1", value="v1", score=0.1),
                            Label(key="k1", value="v2", score=0.9),
                        ],
                    )
                ],
            ),
        ],
        preds_model_class=models.Prediction,
        preds_expected_number=5,
        expected_labels_tuples={
            ("k1", "v1"),
            ("k1", "v2"),
            ("k2", "v6"),
            ("k1", "v2"),
        },
        expected_scores={0.6, 0.4, 1.0, 0.1, 0.9},
        db=db,
    )


def test_add_prediction(
    client: Client,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
    img1: Datum,
    model_name: str,
    dataset_name: str,
    db: Session,
):
    dataset = Dataset.create(dataset_name)
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)

    model = Model.create(model_name)

    # make sure we get an error when passing a non-Prediction object to add_prediction
    with pytest.raises(TypeError):
        model.add_prediction(dataset, "not_a_pred")  # type: ignore

    for pd in pred_dets:
        model.add_prediction(dataset, pd)

    # check we get an error since the dataset has not been finalized
    with pytest.raises(ClientException) as exc_info:
        model.finalize_inferences(dataset)
    assert "DatasetNotFinalizedError" in str(exc_info)

    dataset.finalize()
    model.finalize_inferences(dataset)

    # test get predictions
    pred = model.get_prediction(dataset, img1)
    assert pred
    assert pred.annotations == pred_dets[0].annotations

    client.delete_dataset(dataset_name, timeout=30)


def test_add_empty_prediction(
    client: Client,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
    img1: Datum,
    model_name: str,
    dataset_name: str,
    db: Session,
):
    extra_datum = Datum(uid="some_extra_datum")

    dataset = Dataset.create(dataset_name)
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.add_groundtruth(
        GroundTruth(
            datum=extra_datum,
            annotations=[
                Annotation(
                    labels=[Label(key="k1", value="v1")],
                )
            ],
        )
    )
    dataset.finalize()

    model = Model.create(model_name)

    # make sure we get an error when passing a non-Prediction object to add_prediction
    with pytest.raises(TypeError):
        model.add_prediction(dataset, "not_a_pred")  # type: ignore

    # ensure that adding an empty prediction results in no errors or warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.add_prediction(
            dataset, Prediction(datum=extra_datum, annotations=[])
        )

    for pd in pred_dets:
        model.add_prediction(dataset, pd)
    model.finalize_inferences(dataset)

    # test get predictions
    pred = model.get_prediction(dataset, extra_datum)
    assert pred
    assert len(pred.annotations) == 1

    client.delete_dataset(dataset_name, timeout=30)


def test_add_skipped_prediction(
    client: Client,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
    img1: Datum,
    model_name: str,
    dataset_name: str,
    db: Session,
):
    extra_datum = Datum(uid="some_extra_datum")

    dataset = Dataset.create(dataset_name)
    dataset.add_groundtruth(
        GroundTruth(
            datum=extra_datum,
            annotations=[
                Annotation(
                    labels=[Label(key="k1", value="v1")],
                )
            ],
        )
    )
    dataset.finalize()

    model = Model.create(model_name)
    model.finalize_inferences(dataset)

    # test get predictions
    pred = model.get_prediction(dataset, extra_datum)
    assert pred
    assert len(pred.annotations) == 1

    client.delete_dataset(dataset_name, timeout=30)


def test_validate_model(client: Client, model_name: str):
    with pytest.raises(TypeError):
        Model.create(name=123)  # type: ignore


def test_get_prediction(client: Client, model_name: str, dataset_name: str):
    dataset = Dataset.create(dataset_name)
    model = Model.create(model_name)

    datum = Datum(uid="uid1")
    dataset.add_groundtruth(GroundTruth(datum=datum, annotations=[]))
    dataset.finalize()

    # add a prediction with no annotaitons and check we get a prediction back
    model.add_prediction(dataset, Prediction(datum=datum, annotations=[]))
    assert model.get_prediction(dataset, datum) is not None
