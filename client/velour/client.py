import io
import math
import os
from base64 import b64decode, b64encode
from dataclasses import asdict
from enum import Enum
from typing import Dict, List, Union
from urllib.parse import urljoin

import numpy as np
import PIL.Image
import requests
from tqdm.auto import tqdm

from velour.data_types import (
    BoundingBox,
    BoundingPolygon,
    GroundTruthDetection,
    GroundTruthImageClassification,
    GroundTruthInstanceSegmentation,
    GroundTruthSemanticSegmentation,
    Image,
    Label,
    Point,
    PolygonWithHole,
    PredictedDetection,
    PredictedImageClassification,
    _GroundTruthSegmentation,
    _PredictedSegmentation,
)
from velour.metrics import Task


class DatumTypes(Enum):
    IMAGE = "Image"
    TABULAR = "Tabular"


def _mask_array_to_pil_base64(mask: np.ndarray) -> str:
    f = io.BytesIO()
    PIL.Image.fromarray(mask).save(f, format="PNG")
    f.seek(0)
    mask_bytes = f.read()
    f.close()
    return b64encode(mask_bytes).decode()


def _payload_for_bounding_polygon(poly: BoundingPolygon) -> List[List[int]]:
    """For converting a BoundingPolygon to list of list of ints expected
    by the backend servce
    """
    return [[pt.x, pt.y] for pt in poly.points]


def _list_of_list_to_bounding_polygon(points: List[List[int]]):
    """Inverse of the above method"""
    # backend should return a polygon with the same first and last entries
    # but probably should change the backend to omit the last entry
    # instead of doing it here
    if points[0] != points[-1]:
        raise ValueError("Expected points[0] == points[-1]")

    return BoundingPolygon(points=[Point(*pt) for pt in points[:-1]])


def _payload_for_polys_with_holes(
    polys_with_holes: List[PolygonWithHole],
) -> List[Dict[str, List[List[int]]]]:
    return [
        {
            "polygon": _payload_for_bounding_polygon(sh.polygon),
            "hole": _payload_for_bounding_polygon(sh.hole)
            if sh.hole is not None
            else None,
        }
        for sh in polys_with_holes
    ]


def _det_to_dict(det: Union[PredictedDetection, GroundTruthDetection]) -> dict:
    labels_key = (
        "scored_labels" if isinstance(det, PredictedDetection) else "labels"
    )
    ret = {
        labels_key: [asdict(label) for label in getattr(det, labels_key)],
        "image": asdict(det.image),
    }
    if det.is_bbox:
        ret["bbox"] = [
            det.bbox.xmin,
            det.bbox.ymin,
            det.bbox.xmax,
            det.bbox.ymax,
        ]
    else:
        ret["boundary"] = _payload_for_bounding_polygon(det.boundary)

    return ret


class ClientException(Exception):
    pass


class DatasetBase:
    def __init__(self, client: "Client", name: str):
        self.client = client
        self.name = name

    def get_labels(self) -> List[Label]:
        labels = self.client._requests_get_rel_host(
            f"datasets/{self.name}/labels"
        ).json()

        return [
            Label(key=label["key"], value=label["value"]) for label in labels
        ]

    def finalize(self):
        return self.client._requests_put_rel_host(
            f"datasets/{self.name}/finalize"
        )

    def delete(self):
        return self.client.delete_dataset(self.name)


class TabularDataset(DatasetBase):
    def add_groundtruth(groundtruth: Union[List[Label], Dict[str, Label]]):
        pass


class ImageDataset(DatasetBase):
    def _generate_chunks(
        self,
        data: list,
        chunk_size=100,
        progress_bar_title: str = "Chunking",
        show_progress_bar: bool = True,
    ):
        progress_bar = tqdm(
            total=len(data),
            unit="samples",
            unit_scale=True,
            desc=f"{progress_bar_title} ({self.name})",
            disable=not show_progress_bar,
        )

        number_of_chunks = math.floor(len(data) / chunk_size)
        remainder = len(data) % chunk_size

        for i in range(0, number_of_chunks):
            progress_bar.update(chunk_size)
            yield data[i * chunk_size : (i + 1) * chunk_size]

        if remainder > 0:
            progress_bar.update(remainder)
            yield data[-remainder:]

        progress_bar.close()

    def add_groundtruth(
        self,
        groundtruth: list,
        chunk_size: int = 1000,
        show_progress_bar: bool = True,
    ):
        log = []

        if not isinstance(groundtruth, list):
            raise ValueError("GroundTruth argument should be a list.")

        if len(groundtruth) == 0:
            raise ValueError("Empty list.")

        for chunk in self._generate_chunks(
            groundtruth,
            chunk_size=chunk_size,
            progress_bar_title="Uploading",
            show_progress_bar=show_progress_bar,
        ):
            # Image Classification
            if isinstance(chunk[0], GroundTruthImageClassification):
                payload = {
                    "dataset_name": self.name,
                    "classifications": [
                        {
                            "labels": [asdict(label) for label in clf.labels],
                            "datum": asdict(clf.image),
                        }
                        for clf in chunk
                    ],
                }

                resp = self.client._requests_post_rel_host(
                    "groundtruth-classifications", json=payload
                )

                log += resp

            # Image Segmentation (Semantic, Instance)
            elif isinstance(chunk[0], _GroundTruthSegmentation):

                def _shape_value(
                    shape: Union[List[PolygonWithHole], np.ndarray]
                ):
                    if isinstance(shape, np.ndarray):
                        return _mask_array_to_pil_base64(shape)
                    else:
                        return _payload_for_polys_with_holes(shape)

                payload = {
                    "dataset_name": self.name,
                    "segmentations": [
                        {
                            "shape": _shape_value(seg.shape),
                            "labels": [asdict(label) for label in seg.labels],
                            "image": asdict(seg.image),
                            "is_instance": seg._is_instance,
                        }
                        for seg in chunk
                    ],
                }

                resp = self.client._requests_post_rel_host(
                    "groundtruth-segmentations", json=payload
                )

                log += resp

            # Object Detection
            elif isinstance(chunk[0], GroundTruthDetection):
                payload = {
                    "dataset_name": self.name,
                    "detections": [_det_to_dict(det) for det in chunk],
                }

                resp = self.client._requests_post_rel_host(
                    "groundtruth-detections", json=payload
                )

                log += resp

            # Unknown type.
            else:
                raise NotImplementedError(
                    f"Received groundtruth with type: '{type(chunk[0])}', which is not currently implemented."
                )

        return log

    def get_groundtruth_detections(
        self, image_uid: str
    ) -> List[GroundTruthDetection]:
        resp = self.client._requests_get_rel_host(
            f"datasets/{self.name}/images/{image_uid}/detections"
        ).json()

        def _process_single_gt(gt: dict):
            labels = [Label(**label) for label in gt["labels"]]
            image = Image(**gt["image"])
            if gt["bbox"] is not None:
                return GroundTruthDetection(
                    image=image, labels=labels, bbox=BoundingBox(*gt["bbox"])
                )
            else:
                return GroundTruthDetection(
                    image=image,
                    labels=labels,
                    boundary=_list_of_list_to_bounding_polygon(gt["boundary"]),
                )

        return [_process_single_gt(gt) for gt in resp]

    def _get_segmentations(
        self, image_uid: str, instance: bool
    ) -> Union[
        GroundTruthSemanticSegmentation, GroundTruthInstanceSegmentation
    ]:
        resp = self.client._requests_get_rel_host(
            f"datasets/{self.name}/images/{image_uid}/{'instance' if instance else 'semantic'}-segmentations"
        ).json()

        def _b64_mask_to_array(b64_mask: str) -> np.ndarray:
            mask = b64decode(b64_mask)
            with io.BytesIO(mask) as f:
                img = PIL.Image.open(f)

                return np.array(img)

        data_cls = (
            GroundTruthInstanceSegmentation
            if instance
            else GroundTruthSemanticSegmentation
        )

        return [
            data_cls(
                shape=_b64_mask_to_array(gt["shape"]),
                labels=[Label(**label) for label in gt["labels"]],
                image=Image(**gt["image"]),
            )
            for gt in resp
        ]

    def get_groundtruth_instance_segmentations(
        self, image_uid: str
    ) -> List[GroundTruthInstanceSegmentation]:
        return self._get_segmentations(image_uid, instance=True)

    def get_groundtruth_semantic_segmentations(
        self, image_uid: str
    ) -> List[GroundTruthSemanticSegmentation]:
        return self._get_segmentations(image_uid, instance=False)

    def get_images(self) -> List[Image]:
        images = self.client._requests_get_rel_host(
            f"datasets/{self.name}/images"
        ).json()

        return [
            Image(
                uid=image["uid"], height=image["height"], width=image["width"]
            )
            for image in images
        ]


class Client:
    """Client for interacting with the velour backend"""

    def __init__(self, host: str, access_token: str = None):
        """
        Parameters
        ----------
        host
            the host to connect to. Should start with "http://" or "https://"
        access_token
            the access token if the host requires authentication
        """
        if not (host.startswith("http://") or host.startswith("https://")):
            raise ValueError(
                f"host must stat with 'http://' or 'https://' but got {host}"
            )

        if not host.endswith("/"):
            host += "/"
        self.host = host
        self.access_token = os.getenv("VELOUR_ACCESS_TOKEN", access_token)

        # check the connection by hitting the users endpoint
        email = self._get_users_email()
        success_str = f"Succesfully connected to {self.host}"
        if email is None:
            print(f"{success_str}.")
        else:
            print(f"{success_str} with user {email}.")

    def _get_users_email(self) -> Union[str, None]:
        """Gets the users e-mail address (in the case when auth is enabled)
        or returns None in the case of a no-auth backend.
        """
        resp = self._requests_get_rel_host("user").json()
        return resp["email"]

    def _requests_wrapper(
        self, method_name: str, endpoint: str, *args, **kwargs
    ):
        assert method_name in ["get", "post", "put", "delete"]

        url = urljoin(self.host, endpoint)
        requests_method = getattr(requests, method_name)

        if self.access_token is not None:
            headers = {"Authorization": f"Bearer {self.access_token}"}
        else:
            headers = None
        resp = requests_method(url, headers=headers, *args, **kwargs)
        if not resp.ok:
            if resp.status_code == 500:
                resp.raise_for_status()
            raise ClientException(resp.json()["detail"])

        return resp

    def _requests_post_rel_host(self, endpoint: str, *args, **kwargs):
        return self._requests_wrapper(
            method_name="post", endpoint=endpoint, *args, **kwargs
        )

    def _requests_get_rel_host(self, endpoint: str, *args, **kwargs):
        return self._requests_wrapper(
            method_name="get", endpoint=endpoint, *args, **kwargs
        )

    def _requests_put_rel_host(self, endpoint: str, *args, **kwargs):
        return self._requests_wrapper(
            method_name="put", endpoint=endpoint, *args, **kwargs
        )

    def _requests_delete_rel_host(self, endpoint: str, *args, **kwargs):
        return self._requests_wrapper(
            method_name="delete", endpoint=endpoint, *args, **kwargs
        )

    def _create_dataset(
        self,
        name: str,
        cls: type,
        type_: DatumTypes,
        href: str = None,
        description: str = None,
    ) -> DatasetBase:
        self._requests_post_rel_host(
            "datasets",
            json={
                "name": name,
                "href": href,
                "description": description,
                "type": type_.value,
            },
        )

        return cls(client=self, name=name)

    def create_image_dataset(
        self, name: str, href: str = None, description: str = None
    ):
        return self._create_dataset(
            name=name,
            cls=ImageDataset,
            type_=DatumTypes.IMAGE,
            href=href,
            description=description,
        )

    def create_tabular_dataset(
        self, name: str, href: str = None, description: str = None
    ):
        return self._create_dataset(
            name=name,
            cls=TabularDataset,
            type_=DatumTypes.TABULAR,
            href=href,
            description=description,
        )

    def delete_dataset(self, name: str) -> None:
        job_id = self._requests_delete_rel_host(f"datasets/{name}").json()
        return Job(client=self, job_id=job_id)

    def get_dataset(self, name: str) -> DatasetBase:
        resp = self._requests_get_rel_host(f"datasets/{name}")
        if resp["type"] == DatumTypes.IMAGE:
            class_ = ImageDataset
        elif resp["type"] == DatumTypes.TABULAR:
            class_ = TabularDataset
        else:
            raise RuntimeError(f"Got unexpected type: {resp['type']}")
        return class_(client=self, name=resp.json()["name"])

    def get_datasets(self) -> List[dict]:
        return self._requests_get_rel_host("datasets").json()

    def create_model(
        self, name: str, href: str = None, description: str = None
    ) -> "Model":
        self._requests_post_rel_host(
            "models",
            json={"name": name, "href": href, "description": description},
        )

        return Model(client=self, name=name)

    def delete_model(self, name: str) -> None:
        self._requests_delete_rel_host(f"models/{name}")

    def get_model(self, name: str) -> "Model":
        resp = self._requests_get_rel_host(f"models/{name}")
        return Model(client=self, name=resp.json()["name"])

    def get_models(self) -> List[dict]:
        return self._requests_get_rel_host("models").json()

    def get_all_labels(self) -> List[Label]:
        return self._requests_get_rel_host("labels").json()


class Job:
    def __init__(
        self,
        client: Client,
        job_id: str,
        **kwargs,
    ):
        self._id = job_id
        self.client = client

        for k, v in kwargs.items():
            setattr(self, k, v)

    def status(self) -> str:
        resp = self.client._requests_get_rel_host(f"/jobs/{self._id}").json()
        return resp["status"]


class EvalJob(Job):
    def metrics(self) -> List[dict]:
        return self.client._requests_get_rel_host(
            f"/jobs/{self._id}/metrics"
        ).json()

    def confusion_matrices(self) -> List[dict]:
        return self.client._requests_get_rel_host(
            f"/jobs/{self._id}/confusion-matrices"
        ).json()

    # TODO: replace value with a dataclass?
    def settings(self) -> dict:
        return self.client._requests_get_rel_host(
            f"/jobs/{self._id}/settings"
        ).json()


class Model:
    def __init__(self, client: Client, name: str):
        self.client = client
        self.name = name

    def add_predicted_detections(
        self, dataset: ImageDataset, dets: List[PredictedDetection]
    ) -> None:
        payload = {
            "model_name": self.name,
            "dataset_name": dataset.name,
            "detections": [_det_to_dict(det) for det in dets],
        }

        resp = self.client._requests_post_rel_host(
            "predicted-detections", json=payload
        )

        return resp.json()

    def add_predicted_segmentations(
        self, dataset: ImageDataset, segs: List[_PredictedSegmentation]
    ) -> None:
        payload = {
            "model_name": self.name,
            "dataset_name": dataset.name,
            "segmentations": [
                {
                    "base64_mask": _mask_array_to_pil_base64(seg.mask),
                    "scored_labels": [
                        asdict(scored_label)
                        for scored_label in seg.scored_labels
                    ],
                    "image": asdict(seg.image),
                    "is_instance": seg._is_instance,
                }
                for seg in segs
            ],
        }

        resp = self.client._requests_post_rel_host(
            "predicted-segmentations", json=payload
        )

        return resp.json()

    def add_predicted_classifications(
        self, dataset: DatasetBase, clfs: List[PredictedImageClassification]
    ) -> None:
        payload = {
            "model_name": self.name,
            "dataset_name": dataset.name,
            "classifications": [
                {
                    "scored_labels": [
                        asdict(scored_label)
                        for scored_label in clf.scored_labels
                    ],
                    "datum": asdict(clf.image),
                }
                for clf in clfs
            ],
        }

        resp = self.client._requests_post_rel_host(
            "predicted-classifications", json=payload
        )

        return resp.json()

    def finalize_inferences(self, dataset: DatasetBase) -> None:
        return self.client._requests_put_rel_host(
            f"models/{self.name}/inferences/{dataset.name}/finalize"
        ).json()

    def evaluate_ap(
        self,
        dataset: ImageDataset,
        model_pred_task_type: Task = None,
        dataset_gt_task_type: Task = None,
        iou_thresholds: List[float] = None,
        ious_to_keep: List[float] = None,
        min_area: float = None,
        max_area: float = None,
    ) -> "EvalJob":
        payload = {
            "settings": {
                "model_name": self.name,
                "dataset_name": dataset.name,
                "model_pred_task_type": model_pred_task_type.value
                if model_pred_task_type is not None
                else None,
                "dataset_gt_task_type": dataset_gt_task_type.value
                if dataset_gt_task_type is not None
                else None,
                "min_area": min_area,
                "max_area": max_area,
            }
        }

        if iou_thresholds is not None:
            payload["iou_thresholds"] = iou_thresholds
        if ious_to_keep is not None:
            payload["ious_to_keep"] = ious_to_keep

        resp = self.client._requests_post_rel_host(
            "/ap-metrics", json=payload
        ).json()
        # resp should have keys "missing_pred_labels", "ignored_pred_labels", with values
        # list of label dicts. convert label dicts to Label objects
        for k in ["missing_pred_labels", "ignored_pred_labels"]:
            resp[k] = [Label(**la) for la in resp[k]]

        return EvalJob(client=self.client, **resp)

    def evaluate_classification(self, dataset: DatasetBase) -> EvalJob:
        payload = {
            "settings": {
                "model_name": self.name,
                "dataset_name": dataset.name,
            }
        }

        resp = self.client._requests_post_rel_host(
            "/clf-metrics", json=payload
        ).json()

        return EvalJob(client=self.client, **resp)
