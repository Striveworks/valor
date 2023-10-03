import cProfile
import os
import pickle
import pstats
import timeit
import tracemalloc
from io import BytesIO

import docker

# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import PIL.Image
import requests

from velour.client import Client
from velour.client import Dataset as VelourDataset
from velour.data_generation import generate_segmentation_data
from velour.schemas import ImageMetadata

DATASET_NAME = "profiling"
CHARIOT_HOST = "https://demo01.chariot.striveworks.us/"
LOCAL_HOST = "http://localhost:8000"
ULTRALYTICS_MODEL_NAME = "yolov8n-seg"
CHARIOT_PROJECT_NAME = "n.lind"
CHARIOT_MODEL_NAME = "fasterrcnnresnet-50fpn"


# %%
def setup_database(
    client: Client,
    dataset_name: str,
    n_images: int,
    n_annotations: int,
    n_labels: int,
) -> VelourDataset:
    dataset = generate_segmentation_data(
        client=client,
        dataset_name=dataset_name,
        n_images=n_images,
        n_annotations=n_annotations,
        n_labels=n_labels,
    )

    dataset.finalize()

    return dataset


def _generate_cprofile(fn: callable, **kwargs):
    pr = cProfile.Profile()
    pr.enable()

    fn(**kwargs)

    pr.disable()
    filename = f"profiles/{callable.__name__}_{str(kwargs)}.cprofile"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pr.dump_stats(filename)


def _profile_tracemalloc(fn, output, top_n_traces: int = 10, **kwargs):
    tracemalloc.start()
    fn(**kwargs)
    snapshot = (
        tracemalloc.take_snapshot()
        .filter_traces(
            (tracemalloc.Filter(True, f"{os.getcwd()}/*", all_frames=True),)
        )
        .statistics("lineno")
    )
    tracemalloc_output = dict()
    for i, stat in enumerate(snapshot[:top_n_traces]):
        tracemalloc_output.update(
            {
                i: {
                    "filename": stat.traceback._frames[0][0],
                    "line": stat.traceback._frames[0][1],
                    "size": stat.size,
                    "count": stat.count,
                }
            }
        )

    output.update(tracemalloc_output)


def _profile_func(
    fn: callable,
    top_n_traces: int = 10,
    **kwargs,
) -> dict:
    print(f"Profiling {fn.__name__} with args {kwargs}")

    success = False
    exception = " "
    output = {}
    start = timeit.default_timer()

    try:
        _profile_tracemalloc(
            fn=fn, output=output, top_n_traces=top_n_traces, **kwargs
        )
        success = True
    except Exception as e:
        exception = str(e)

    stop = timeit.default_timer()
    timeit_output = {
        "start": start,
        "total_runtime_seconds": round(stop - start, 2),
    }
    if success:
        print(f"Succeeded in {stop-start} seconds")
    else:
        print(f"Failed in {stop-start} seconds with error {exception}")

    return kwargs | timeit_output | output | {"exception": exception}


def profile_velour(
    client: Client,
    dataset_name: str,
    n_image_grid: list,
    n_annotation_grid: list,
    n_label_grid: list,
    using_docker: bool = True,
    db_container_name: str = "velour_db_1",
    service_container_name: str = "velour_service_1",
) -> list:
    if using_docker:
        docker_client = docker.from_env()
        db_container = docker_client.containers.get(db_container_name)
        service_container = docker_client.containers.get(
            service_container_name
        )

        # restart containerss to clear extraneous objects
        # db_container.restart()
        # service_container.restart()

    output = list()
    for n_images in n_image_grid:
        for n_annotations in n_annotation_grid:
            for n_labels in n_label_grid:
                kwargs = {
                    "client": client,
                    "dataset_name": dataset_name,
                    "n_images": n_images,
                    "n_annotations": n_annotations,
                    "n_labels": n_labels,
                }

                results = _profile_func(setup_database, **kwargs)

                if using_docker:
                    service_stats = get_container_stats(
                        container=service_container,
                        indicator="service",
                    )
                    db_stats = get_container_stats(
                        container=db_container,
                        indicator="db",
                    )
                    results = results | service_stats | db_stats

                output.append(results)

                # create checkpoint
                filename = f"{os.getcwd()}/utils/profiles/{dataset_name}.pkl"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(
                    filename,
                    "wb+",
                ) as f:
                    pickle.dump(output, f)

    return output


def load_results_from_disk(
    dataset_name: str,
):
    with open(f"{os.getcwd()}/utils/profiles/{dataset_name}.pkl", "rb") as f:
        output = pickle.load(f)

    df = pd.DataFrame.from_records(output)

    return df


# TODO type
def get_container_stats(container, indicator: str = "") -> dict:
    stats = container.stats(stream=False)
    return {
        f"{indicator}memory_usage": stats["memory_stats"]["usage"],
        f"{indicator}memory_limit": stats["memory_stats"]["limit"],
        f"{indicator}cpu_usage": stats["cpu_stats"]["cpu_usage"][
            "total_usage"
        ],
        f"{indicator}cpu_usage_kernel": stats["cpu_stats"]["cpu_usage"][
            "usage_in_kernelmode"
        ],
        f"{indicator}cpu_usage_user": stats["cpu_stats"]["cpu_usage"][
            "usage_in_usermode"
        ],
        f"{indicator}cpu_usage_system": stats["cpu_stats"]["system_cpu_usage"],
        f"{indicator}cpu_throttled_time": stats["cpu_stats"][
            "throttling_data"
        ]["throttled_time"],
    }


client = Client(LOCAL_HOST)

results = profile_velour(
    client=client,
    dataset_name="profiling",
    n_image_grid=[2, 4, 5],
    n_annotation_grid=[5],
    n_label_grid=[2],
)

# %%


# %%
def download_image(img: ImageMetadata, image_url_dict: dict) -> PIL.Image:
    url = image_url_dict[img.uid]
    img_data = BytesIO(requests.get(url).content)
    return PIL.Image.open(img_data)


def get_profile_stats(name: str, number_of_images: int):
    stats = (
        pstats.Stats(f"profiles/{name}_{number_of_images}_images.cprofile")
        .strip_dirs()
        .sort_stats(1)
    )

    return stats


# %%
# Graveyard


# def run_ultralytics_inference(
#     client: Client, dataset: VelourDataset, image_url_dict: dict
# ) -> VelourModel:
#     # reset
#     client.delete_model(ULTRALYTICS_MODEL_NAME)

#     # create new model
#     try:
#         velour_yolo_model = VelourModel.create(client, ULTRALYTICS_MODEL_NAME)
#     except:
#         velour_yolo_model = VelourModel.get(client, ULTRALYTICS_MODEL_NAME)

#     yolo_model = YOLO(f"{ULTRALYTICS_MODEL_NAME}.pt")

#     for image_metadata in dataset.get_images():
#         # retrieve image
#         image = download_image(
#             img=image_metadata, image_url_dict=image_url_dict
#         )

#         # YOLO inference
#         results = yolo_model(image, verbose=False)

#         # convert YOLO result into Velour prediction
#         prediction = parse_yolo_object_detection(
#             results[0], image=image_metadata, label_key="name"
#         )

#         # add prediction to the model
#         velour_yolo_model.add_prediction(prediction)

#     velour_yolo_model.finalize_inferences(dataset)

#     return velour_yolo_model


# def run_chariot_inference(
#     client: Client, dataset: VelourDataset, image_url_dict: dict
# ) -> VelourModel:
#     chariot_model = ChariotModel(
#         name=CHARIOT_MODEL_NAME, project_name=CHARIOT_PROJECT_NAME
#     )

#     # reset
#     client.delete_model(chariot_model.id)

#     # create new Velour model
#     (
#         velour_chariot_model,
#         velour_chariot_parser,
#     ) = get_chariot_model_integration(client, chariot_model, "detect")

#     for image_metadata in dataset.get_images():
#         # retrieve image
#         image = download_image(
#             img=image_metadata, image_url_dict=image_url_dict
#         )

#         # Chariot Inference
#         result = chariot_model.detect(image)

#         # convert Chariot result into Velour prediction
#         prediction = velour_chariot_parser(
#             datum=image_metadata.to_datum(),
#             result=result,
#         )

#         # add prediction to the model
#         velour_chariot_model.add_prediction(prediction)

#     velour_chariot_model.finalize_inferences(dataset)

#     return velour_chariot_model


# def run_workflow(number_of_images: int):
#     client = Client(LOCAL_HOST)

#     velour_coco_dataset, annotations = run_setup(
#         client=client, number_of_images=number_of_images
#     )
#     image_url_dict = {
#         str(img_dict["id"]): img_dict["coco_url"]
#         for img_dict in annotations["images"]
#     }
#     run_ultralytics_inference(
#         client=client,
#         dataset=velour_coco_dataset,
#         image_url_dict=image_url_dict,
#     )


# %%


# def plot_results(output: pd.DataFrame):
#     df = pd.DataFrame(output)

#     # set up the figure and axes
#     fig = plt.figure(figsize=(8, 3))
#     ax1 = fig.add_subplot(121, projection="3d")

#     # fake data
#     # x = df["number_of_images"]
#     # y = df["number_of_annotations"]
#     # z = df["total_runtime_seconds"]

#     x = [0, 1, 2, 3]
#     y = [0, 1, 2, 3, 4, 5]
#     z = [0, 1, 2, 3, 4, 5]

#     # x = np.arange(-5, 5, 0.25)
#     # y = np.arange(-5, 5, 0.25)
#     x, y = np.meshgrid(x, y)
#     z = x**2 + y**2

#     print(z)

#     ax1.plot_surface(
#         x,
#         y,
#         z,
#         cmap=cm.coolwarm,
#         linewidth=0,
#     )
#     ax1.set_title("Runtime (seconds)")
#     ax1.set_xlabel("Number of Images")
#     ax1.set_ylabel("Number of Annotations")

#     plt.show()


# plot_results(output)


# # %%[markdown]
# # Graveyard: finding distributions of images and annotations
# # %%
# import pandas as pd

# pd.DataFrame(ex)
# # %%

# with open("./coco/annotations/panoptic_train2017.json") as f:
#     metadata = json.load(f)

# annotations = metadata["annotations"]
# # get the count of annotations per image
# for i in metadata["images"][:5]:
#     print(i["id"])

# df = pd.DataFrame(annotations)
# df["segments_info"].str.len().plot.hist()
# # %%

# len(metadata["images"])
# # %%
