# Reference: https://github.com/Striveworks/chariot/blob/main/py/libs/sdk/integration_tests/test_datasets_cv.py
# import json
# import os
# import random
# import tempfile
# import threading
# import time
# from datetime import datetime, timedelta
# from io import BytesIO

# import numpy as np
# import pytest
# from exif import Image as ExifImage
# from PIL import Image

# chariot = pytest.importorskip("chariot")
# from chariot.client import connect
# from chariot.datasets import (
#     Dataset,
#     DatasetDoesNotExistError,
#     DatasetVersion,
#     DegreesMinutesSeconds,
#     DisjointFilterDatumMetadataSet,
#     FilterDatumMetadata,
#     FilterLabel,
#     HorizontalVersionPayload,
#     JointFilterDatumMetadataSet,
#     SpatialWindowFilter,
#     TemporalWindowFilter,
#     VerticalVersionPayload,
#     create_horizontal_version,
#     create_vertical_version,
#     get_datasets_in_project,
#     rescan_bucket,
#     upload_annotated_data,
#     upload_archive_to_existing_dataset,
#     upload_archives_as_dataset,
#     upload_dataset_from_folder,
#     upload_dataset_from_imagefolder,
#     upload_datums_to_bucket,
#     wait_for_dataset_uploads,
# )

# @pytest.fixture
# def img_size():
#     return (50, 50, 3)

# @pytest.fixture
# def fake_image(img_size) -> Image.Image:
#     return Image.fromarray(np.random.randint(255, size=img_size, dtype=np.uint8))

# def jitter(n: float) -> float:
#     return n + random.uniform(-1, 1)

# def datetime_between(earlier: datetime, later: datetime) -> str:
#     """Create a random datetime between two datetimes"""
#     delta = later - earlier
#     delta_seconds = int(delta.total_seconds())
#     in_between = random.randint(0, delta_seconds)
#     new_datetime = earlier + timedelta(seconds=in_between)
#     return new_datetime.strftime("%Y:%m:%d, %H:%M:%S")

# def add_exif_tags(image: Image) -> ExifImage:
#     """
#     Assigns exif data to an image
#     """
#     austin = {
#         "gps_latitude": (jitter(30.0), jitter(15.0), jitter(59.9976)),
#         "gps_latitude_ref": "N",
#         "gps_longitude": (jitter(97.0), jitter(43.0), jitter(59.9880)),
#         "gps_longitude_ref": "W",
#     }
#     boston = {
#         "gps_latitude": (jitter(42.0), jitter(21.0), jitter(40.1220)),
#         "gps_latitude_ref": "N",
#         "gps_longitude": (jitter(71.0), jitter(3.0), jitter(25.4988)),
#         "gps_longitude_ref": "W",
#     }

#     _bytes = BytesIO()
#     image.save(_bytes, format="jpeg")

#     image = ExifImage(_bytes.getvalue())

#     location = austin if random.random() <= 0.5 else boston
#     dt = datetime_between(
#         earlier=datetime(2022, 1, 1, 1, 1, 1, 1),
#         later=datetime(2023, 1, 1, 1, 1, 1, 1),
#     )

#     image.gps_latitude = location["gps_latitude"]
#     image.gps_latitude_ref = location["gps_latitude_ref"]
#     image.gps_longitude = location["gps_longitude"]
#     image.gps_longitude_ref = location["gps_longitude_ref"]
#     image.datetime_original = dt

#     return image

# def create_fake_image_annotated_dataset(num_images=12, seq=0):
#     dataset_name = (
#         time.strftime("%Y%m%d-%H%M%S")
#         + f"-{seq}"
#         + "-create_fake_image_ann_dataset"
#         + f"-{threading.get_ident()}"
#     )

#     with tempfile.TemporaryDirectory() as tempdir:
#         train_annotations = []
#         val_annotations = []
#         fake_imgs = []
#         labels = ["a", "b"]
#         for i in range(num_images):
#             path = os.path.join(tempdir, f"{i}.jpg")

#             fake_img = fake_image(img_size)
#             image = add_exif_tags(fake_img)
#             with open(path, "wb") as f:
#                 f.write(image.get_file())
#             fake_imgs.append(Image.open(path))

#             annotation = {
#                 "path": path,
#                 "annotations": [{"class_label": labels[i % 2]}],
#             }
#             if i % 2 == 0:
#                 train_annotations.append(annotation)
#             else:
#                 val_annotations.append(annotation)

#         train_ann_file = os.path.join(tempdir, "train.jsonl")
#         val_ann_file = os.path.join(tempdir, "val.jsonl")
#         for fname, annotations in [
#             (train_ann_file, train_annotations),
#             (val_ann_file, val_annotations),
#         ]:
#             with open(fname, "w") as f:
#                 f.writelines([json.dumps(ann) + "\n" for ann in annotations])

#         upload_annotated_data(
#             description="for e2e test (ann)",
#             name=dataset_name,
#             project_id=get_test_project_id(),
#             dataset_type="image",
#             train_annotation_file=train_ann_file,
#             val_annotation_file=val_ann_file,
#         )

#     return dataset_name, fake_imgs

# def test_chariot_ds_to_velour_ds():

#     connect(host="https://production.chariot.striveworks.us")

#     name, _ = create_fake_image_annotated_dataset()

#     ds = Dataset(get_test_project_id(), ds_id)
#         ds.delete()
