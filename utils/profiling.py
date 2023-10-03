# %%
import cProfile
import pickle
import pstats
import timeit
import tracemalloc
from datetime import datetime
from io import BytesIO

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
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


def _profile_func(fn: callable, dump: bool = True, **kwargs) -> dict:
    dt = datetime.now.strftime("%d/%m/%Y_%H:%M:%S")
    print(f"Profiling {fn.__name__} with args {kwargs} at {dt}")

    pr = cProfile.Profile()
    pr.enable()
    tracemalloc.start()
    start = timeit.default_timer()

    success = False
    exception = " "
    try:
        fn(**kwargs)
        success = True
    except Exception as e:
        exception = str(e)

    stop = timeit.default_timer()
    tracemalloc.take_snapshot()
    pr.disable()

    if success:
        print(f"Succeeded in {stop-start} seconds")
    else:
        print(f"Failed in {stop-start} seconds with error {exception}")

    if dump:
        pr.dump_stats(f"profiles/{callable.__name__}_{dt}.cprofile")

    return kwargs | {
        "total_runtime_seconds": stop - start,
        "min_memory_usage": 0,
        "max_memory_usage": 0,
    }


def profile_velour(
    client: Client,
    dataset_name: str,
    n_image_grid: list,
    n_annotation_grid: list,
    n_label_grid: list,
):
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
                output.append(results)

                # create checkpoint
                with open(
                    f"profiles/{n_images}_images_{n_annotations}_annotations_{n_labels}_labels.pkl",
                    "wb",
                ) as f:
                    pickle.dump(output, f)

    return output


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


def load_results_from_disk(
    n_images: int,
    n_annotations: int,
):
    with open(
        f"profiles/{n_images}_images_{n_annotations}_annotations.pkl", "rb"
    ) as f:
        output = pickle.load(f)

    return output


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
