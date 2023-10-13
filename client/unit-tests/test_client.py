import pytest

from velour.client import Client, ClientException

dset_name = "test_dataset"
model_name = "test_model"


@pytest.fixture
def client():
    return Client(host="http://localhost:8000")


def test_client():
    bad_url = "localhost:8000"

    with pytest.raises(ValueError):
        Client(host=bad_url)

    bad_url2 = "http://localhost:8111"

    with pytest.raises(Exception):
        Client(host=bad_url2)

    good_url = "http://localhost:8000"

    assert Client(host=good_url)


def test__requests_wrapper(client: Client):
    with pytest.raises(ValueError):
        client._requests_wrapper("get", "/datasets/fake_dataset/status")

    with pytest.raises(AssertionError):
        client._requests_wrapper("bad_method", "datasets/fake_dataset/status")

    with pytest.raises(ClientException):
        client._requests_wrapper("get", "not_an_endpoint")

    with pytest.raises(ClientException):
        client._requests_wrapper("get", "datasets/fake_dataset/status")

    # NOTE: _requests_wrapper successes are implicitly tested in test_main.py and test_cred.py


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
