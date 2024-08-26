import pandas as pd
import pytest
from valor_core import enums
from valor_core.utilities import (
    create_validated_groundtruth_df,
    create_validated_prediction_df,
)


def test_create_validated_groundtruth_df():

    # test that the dataframe has the right columns
    df = pd.DataFrame(
        [
            {
                "datum_uid": "uid0",
                "datum_id": "img0",
                "id": "gt0",
                "label_key": "class_label",
                "label_value": "dog",
            },
            {
                "datum_uid": "uid1",
                "datum_id": "img1",
                "id": "gt1",
                "label_key": "class_label",
                "label_value": "dog",
            },
            {
                "datum_uid": "uid2",
                "datum_id": "img2",
                "id": "gt2",
                "label_key": "class_label",
                "label_value": "dog",
            },
            {
                "datum_uid": "uid3",
                "datum_id": "img3",
                "id": "gt3",
                "label_key": "class_label",
                "label_value": "dog",
            },
            {
                "datum_uid": "uid4",
                "datum_id": "img4",
                "id": "gt4",
                "label_key": "class_label",
                "label_value": "dog",
            },
        ]
    )

    with pytest.raises(ValueError):
        create_validated_groundtruth_df(
            df, task_type=enums.TaskType.CLASSIFICATION
        )

    # test that we get an error if we don't pass non-unique IDs
    df = pd.DataFrame(
        [
            {
                "datum_uid": "uid0",
                "datum_id": "img0",
                "id": "gt0",
                "label_key": "class_label",
                "label_value": "dog",
                "annotation_id": 1,
                "label_id": 0,
            },
            {
                "datum_uid": "uid1",
                "datum_id": "img1",
                "id": "gt0",
                "label_key": "class_label",
                "annotation_id": 2,
                "label_value": "dog",
                "label_id": 0,
            },
        ]
    )

    with pytest.raises(ValueError):
        create_validated_groundtruth_df(
            df, task_type=enums.TaskType.CLASSIFICATION
        )

    # test that groundtruth dataframes can't have scores
    df = pd.DataFrame(
        [
            {
                "datum_uid": "uid0",
                "datum_id": "img0",
                "id": "gt0",
                "label_key": "class_label",
                "label_value": "dog",
                "annotation_id": 1,
                "label_id": 0,
                "score": 0.99,
            },
            {
                "datum_uid": "uid1",
                "datum_id": "img1",
                "id": "gt1",
                "label_key": "class_label",
                "annotation_id": 2,
                "label_value": "dog",
                "label_id": 0,
                "score": 0.01,
            },
        ]
    )

    with pytest.raises(ValueError):
        create_validated_groundtruth_df(
            df, task_type=enums.TaskType.CLASSIFICATION
        )

    # test correct example
    df = pd.DataFrame(
        [
            {
                "datum_uid": "uid0",
                "datum_id": "img0",
                "id": "gt0",
                "label_key": "class_label",
                "label_value": "dog",
                "annotation_id": 1,
                "label_id": 0,
            },
            {
                "datum_uid": "uid1",
                "datum_id": "img1",
                "id": "gt1",
                "label_key": "class_label",
                "annotation_id": 2,
                "label_value": "dog",
                "label_id": 0,
            },
        ]
    )

    create_validated_groundtruth_df(
        df, task_type=enums.TaskType.CLASSIFICATION
    )


def test_create_validated_prediction_df():

    # test that the dataframe has the right columns
    df = pd.DataFrame(
        [
            {
                "datum_uid": "uid0",
                "datum_id": "img0",
                "id": "pd0",
                "label_key": "class_label",
                "label_value": "dog",
            },
            {
                "datum_uid": "uid1",
                "datum_id": "img1",
                "id": "pd1",
                "label_key": "class_label",
                "label_value": "dog",
            },
            {
                "datum_uid": "uid2",
                "datum_id": "img2",
                "id": "pd2",
                "label_key": "class_label",
                "label_value": "dog",
            },
            {
                "datum_uid": "uid3",
                "datum_id": "img3",
                "id": "pd3",
                "label_key": "class_label",
                "label_value": "dog",
            },
            {
                "datum_uid": "uid4",
                "datum_id": "img4",
                "id": "pd4",
                "label_key": "class_label",
                "label_value": "dog",
            },
        ]
    )

    with pytest.raises(ValueError):
        create_validated_prediction_df(
            df, task_type=enums.TaskType.CLASSIFICATION
        )

    # test that we get an error if we don't pass non-unique IDs
    df = pd.DataFrame(
        [
            {
                "datum_uid": "uid0",
                "datum_id": "img0",
                "id": "pd0",
                "label_key": "class_label",
                "label_value": "dog",
                "annotation_id": 1,
                "label_id": 0,
                "score": 0.08,
            },
            {
                "datum_uid": "uid1",
                "datum_id": "img1",
                "id": "pd0",
                "label_key": "class_label",
                "annotation_id": 2,
                "label_value": "cat",
                "label_id": 0,
                "score": 0.92,
            },
        ]
    )

    with pytest.raises(ValueError):
        create_validated_prediction_df(
            df, task_type=enums.TaskType.CLASSIFICATION
        )

    # test that we get an error if the prediction scores for a given label key and datum don't add up to 1
    df = pd.DataFrame(
        [
            {
                "datum_uid": "uid0",
                "datum_id": "img0",
                "id": "pd0",
                "label_key": "class_label",
                "label_value": "dog",
                "annotation_id": 1,
                "label_id": 0,
                "score": 0.04,
            },
            {
                "datum_uid": "uid0",
                "datum_id": "img0",
                "id": "pd1",
                "label_key": "class_label",
                "annotation_id": 2,
                "label_value": "cat",
                "label_id": 0,
                "score": 0.92,
            },
        ]
    )

    with pytest.raises(ValueError):
        create_validated_prediction_df(
            df, task_type=enums.TaskType.CLASSIFICATION
        )

    # test correct example
    df = pd.DataFrame(
        [
            {
                "datum_uid": "uid0",
                "datum_id": "img0",
                "id": "pd0",
                "label_key": "class_label",
                "label_value": "dog",
                "annotation_id": 1,
                "label_id": 0,
                "score": 0.08,
            },
            {
                "datum_uid": "uid0",
                "datum_id": "img0",
                "id": "pd1",
                "label_key": "class_label",
                "annotation_id": 2,
                "label_value": "cat",
                "label_id": 0,
                "score": 0.92,
            },
        ]
    )

    create_validated_prediction_df(df, task_type=enums.TaskType.CLASSIFICATION)
