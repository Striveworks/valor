import pytest

from velour.client import Client, ClientException
from velour.integrations.chariot import chariot_ds_to_velour_ds

chariot_client = pytest.importorskip("chariot.client")
chariot_datasets = pytest.importorskip("chariot.datasets.dataset")

# Reference: https://github.com/Striveworks/chariot/blob/main/py/libs/sdk/integration_tests/test_datasets_cv.py

if __name__ == "__main__":

    chariot_client.connect(host="https://production.chariot.striveworks.us")

    # List available datasets in project
    project_name = "Global"
    # project_name = "OnBoarding"
    dataset_list = chariot_datasets.get_datasets_in_project(
        limit=25, offset=0, project_name=project_name
    )

    lookup = {}
    print("Datasets")
    for i in range(len(dataset_list)):
        lookup[str(dataset_list[i].name).strip()] = dataset_list[i]
        print(" " + str(i) + ": " + dataset_list[i].name)

    chariot_ds = lookup["CIFAR-10"]
    # chariot_ds = lookup["Testing"]

    velour_client = Client(host="http://localhost:8000")
    # velour_client = Client(
    #     "https://velour.striveworks.us/api/",
    #     access_token="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik9rLURVZlRlcEczTlpuUjRCaFNhVCJ9.eyJlbWFpbCI6ImMuemFsb29tQHN0cml2ZXdvcmtzLnVzIiwiaXNzIjoiaHR0cHM6Ly92ZWxvdXIudXMuYXV0aDAuY29tLyIsInN1YiI6Imdvb2dsZS1vYXV0aDJ8MTEzNjkyOTUwNjgyMzgxMzkzOTQwIiwiYXVkIjpbImh0dHBzOi8vdmVsb3VyLnN0cml2ZXdvcmtzLnVzLyIsImh0dHBzOi8vdmVsb3VyLnVzLmF1dGgwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE2ODM3MjgzMzgsImV4cCI6MTY4MzgxNDczOCwiYXpwIjoiSkhzTDNXZ0N1ZVd5S2kwbW5CbDhvNDdyMFV4NlhHOFAiLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIn0.0d9dlO6NQETwvTdtFU-pPXFNrUDggqJAi09nhpf64b9TTDnrrhuKrcIfsK5OXTfJIt0lKxWx_Wbl_tfzYeeIGGxiImR07Qm_gl6_oGeKyJkLVYdgWqzoLRVbIMyRrpgs9YUdVHVXu9z-LHCQ4T0dgu0PETqmytoWrWSgO63xCgaAM8t6sGcHwvqYaL6bWSla4VQBdKkfGMTHUypsFopliES8xpTKp6SC8NXf9Ld2RMgOplLSO7hJJmyRXt5YNgPtZvBJZfWV3uLc93rQs55RSgUPW0W0a1qio7CI8YGYx1N0UX3FZFnIAxG85u3XTu_j6KYjiwWnFRbyGoUpsM05RQ",
    # )

    try:
        velour_client.delete_dataset(chariot_ds.name)
    except ClientException as err:
        print(err)

    velour_ds = chariot_ds_to_velour_ds(
        velour_client=velour_client,
        chariot_dataset=chariot_ds,
        chunk_size=1000,
    )

    print(velour_ds.get_labels())

    velour_client.delete_dataset(velour_ds.name)
