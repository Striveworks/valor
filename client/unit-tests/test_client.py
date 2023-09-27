from velour.integrations.coco import _merge_annotation_list
from velour.schemas import (
    Label,
)
from velour.enums import TaskType


def test__merge_annotation_list():
    """Check that we get the correct annotation set after merging semantic segmentions"""

    initial_annotations = [
        dict(
            task_type=TaskType.SEMANTIC_SEGMENTATION,
            labels=set([Label(key="k1", value="v1"), Label(key="k2", value="v2")]),
            mask=[True, False, False],
        ),
        dict(
            task_type=TaskType.SEMANTIC_SEGMENTATION,
            labels=set([Label(key="k1", value="v1"), Label(key="k3", value="v3")]),
            mask=[False, False, True],
        ),
        dict(
            task_type=TaskType.INSTANCE_SEGMENTATION,
            labels=set([Label(key="k1", value="v1"), Label(key="k3", value="v3")]),
            mask=[False, True, False],
        ),
    ]

    expected = [
        dict(
            task_type=TaskType.SEMANTIC_SEGMENTATION,
            labels=set(
                [
                    Label(key="k1", value="v1"),
                    Label(key="k2", value="v2"),
                    Label(key="k3", value="v3"),
                ]
            ),
            mask=[True, False, True],
        ),
        dict(
            task_type=TaskType.INSTANCE_SEGMENTATION,
            labels=set(
                [
                    Label(key="k1", value="v1"),
                    Label(key="k3", value="v3"),
                ]
            ),
            mask=[False, True, False],
        ),
    ]

    label_map = {
        Label(key="k1", value="v1"): [0, 1],
        Label(key="k2", value="v2"): [0],
        Label(key="k3", value="v3"): [1],
    }

    merged_annotations = _merge_annotation_list(
        annotation_list=initial_annotations, label_map=label_map
    )

    for i, v in enumerate(merged_annotations):
        assert (
            merged_annotations[i]["labels"] == expected[i]["labels"]
        ), "Labels didn't merge as expected"
        assert sum(merged_annotations[i]["mask"]) == sum(
            expected[i]["mask"]
        ), "Masks didn't merge as expected"


if __name__ == "__main__":
    test__merge_annotation_list()


# NOTE: Will probably reimplemnt in the future under `velour.client.Evaluation`

# from velour.client import Model


# def test__group_evaluation_settings():
#     eval_settings = [
#         {
#             "model": "model",
#             "dataset": "dset1",
#             "model_pred_task_type": "Classification",
#             "dataset_gt_task_type": "Classification",
#             "id": 1,
#         },
#         {
#             "model": "model",
#             "dataset": "dset3",
#             "model_pred_task_type": "Classification",
#             "dataset_gt_task_type": "Other Task",
#             "id": 2,
#         },
#         {
#             "model": "model",
#             "dataset": "dset2",
#             "model_pred_task_type": "Classification",
#             "dataset_gt_task_type": "Classification",
#             "id": 3,
#         },
#     ]

#     groupings = Model._group_evaluation_settings(eval_settings)

#     assert len(groupings) == 2

#     assert groupings == [
#         {
#             "ids": [1, 3],
#             "settings": {
#                 "model_pred_task_type": "Classification",
#                 "dataset_gt_task_type": "Classification",
#             },
#             "datasets": ["dset1", "dset2"],
#         },
#         {
#             "ids": [2],
#             "settings": {
#                 "model_pred_task_type": "Classification",
#                 "dataset_gt_task_type": "Other Task",
#             },
#             "datasets": ["dset3"],
#         },
#     ]
