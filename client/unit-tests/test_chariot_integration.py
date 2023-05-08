import pytest
from chariot.client import connect
from chariot.datasets.dataset import Dataset, get_datasets_in_project

from velour.convert import chariot_ds_to_velour_ds


@pytest.fixture
def project():
    return "Global"


@pytest.fixture
def image_classification_dataset():
    return "CIFAR-10"


@pytest.fixture
def image_segmentation_dataset():
    return "spacenet_buildings_vegas"


@pytest.fixture
def object_detection_dataset():
    return "Cars Overhead With Context - COWC"


@pytest.fixture
def datasets(project: str):
    connect(host="https://production.chariot.striveworks.us")

    # List available datasets in project
    datasets = get_datasets_in_project(
        limit=25, offset=0, project_name=project
    )
    lookup = {}
    for i in range(len(datasets)):
        lookup[str(datasets[i].name).strip()] = datasets[i]

    return lookup


# Image Classification
def test_chariot_load_image_classification_dataset(
    datasets: dict[Dataset], image_classification_dataset: str
):
    assert image_classification_dataset in datasets
    dsv = datasets[image_classification_dataset].versions[0]
    velour_ds = chariot_ds_to_velour_ds(dsv, "DS1")
    assert len(velour_ds) == dsv.summary.num_classification_labels


# Object Detection
def test_chariot_load_image_segmentation_dataset(
    datasets: dict[Dataset], image_segmentation_dataset: str
):
    assert image_segmentation_dataset in datasets
    dsv = datasets[image_segmentation_dataset].versions[0]
    velour_ds = chariot_ds_to_velour_ds(dsv, "DS2")
    assert len(velour_ds) <= dsv.summary.num_image_datums


# Image Segmentation
def test_chariot_load_object_detection_dataset(
    datasets: dict[Dataset], object_detection_dataset: str
):
    assert object_detection_dataset in datasets
    dsv = datasets[object_detection_dataset].versions[0]
    velour_ds = chariot_ds_to_velour_ds(dsv, "DS3")
    print(dsv.summary)
    print(dsv.supported_task_types)
    assert len(velour_ds) == dsv.summary.num_bounding_boxes


if __name__ == "__main__":

    connect(host="https://production.chariot.striveworks.us")

    # List available datasets in project
    project_name = "Global"
    dataset_list = get_datasets_in_project(
        limit=25, offset=0, project_name=project_name
    )

    dslu = {}
    print("Datasets")
    for i in range(len(dataset_list)):
        dslu[str(dataset_list[i].name).strip()] = dataset_list[i]
        print(" " + str(i) + ": " + dataset_list[i].name)
