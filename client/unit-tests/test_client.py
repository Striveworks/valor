from velour.client import Model


def test__group_evaluation_settings():
    eval_settings = [
        {
            "model": "model",
            "dataset": "dset1",
            "model_pred_task_type": "Classification",
            "dataset_gt_task_type": "Classification",
            "id": 1,
        },
        {
            "model": "model",
            "dataset": "dset3",
            "model_pred_task_type": "Classification",
            "dataset_gt_task_type": "Other Task",
            "id": 2,
        },
        {
            "model": "model",
            "dataset": "dset2",
            "model_pred_task_type": "Classification",
            "dataset_gt_task_type": "Classification",
            "id": 3,
        },
    ]

    groupings = Model._group_evaluation_settings(eval_settings)

    assert len(groupings) == 2

    assert groupings == [
        {
            "ids": [1, 3],
            "settings": {
                "model_pred_task_type": "Classification",
                "dataset_gt_task_type": "Classification",
            },
            "datasets": ["dset1", "dset2"],
        },
        {
            "ids": [2],
            "settings": {
                "model_pred_task_type": "Classification",
                "dataset_gt_task_type": "Other Task",
            },
            "datasets": ["dset3"],
        },
    ]
