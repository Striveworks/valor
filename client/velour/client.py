import io
import os
from base64 import b64encode
from dataclasses import asdict
from typing import Dict, List, Union
from urllib.parse import urljoin

import numpy as np
import requests
from PIL import Image as PILImage

from velour.data_types import (
    BoundingPolygon,
    GroundTruthDetection,
    GroundTruthImageClassification,
    GroundTruthSegmentation,
    Image,
    Label,
    PolygonWithHole,
    PredictedDetection,
    PredictedImageClassification,
    PredictedSegmentation,
)


def _mask_array_to_pil_base64(mask: np.ndarray) -> str:
    f = io.BytesIO()
    PILImage.fromarray(mask).save(f, format="PNG")
    f.seek(0)
    mask_bytes = f.read()
    f.close()
    return b64encode(mask_bytes).decode()


def _payload_for_bounding_polygon(poly: BoundingPolygon) -> List[List[int]]:
    return [[pt.x, pt.y] for pt in poly.points]


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


class ClientException(Exception):
    pass


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

    def upload_groundtruth_detections(
        self, dataset_name: str, dets: List[GroundTruthDetection]
    ) -> List[int]:
        payload = {
            "dataset_name": dataset_name,
            "detections": [
                {
                    "boundary": _payload_for_bounding_polygon(det.boundary),
                    "labels": [asdict(label) for label in det.labels],
                    "image": asdict(det.image),
                }
                for det in dets
            ],
        }

        resp = self._requests_post_rel_host(
            "groundtruth-detections", json=payload
        )
        resp.raise_for_status()

        return resp.json()

    def upload_groundtruth_segmentations(
        self, dataset_name: str, segs: List[GroundTruthSegmentation]
    ) -> List[int]:
        def _shape_value(shape: Union[List[PolygonWithHole], np.ndarray]):
            if isinstance(shape, np.ndarray):
                return _mask_array_to_pil_base64(shape)
            else:
                return _payload_for_polys_with_holes(shape)

        payload = {
            "dataset_name": dataset_name,
            "segmentations": [
                {
                    "shape": _shape_value(seg.shape),
                    "labels": [asdict(label) for label in seg.labels],
                    "image": asdict(seg.image),
                    "is_instance": seg._is_instance,
                }
                for seg in segs
            ],
        }

        resp = self._requests_post_rel_host(
            "groundtruth-segmentations", json=payload
        )
        resp.raise_for_status()

        return resp.json()

    def upload_predicted_detections(
        self, model_name: str, dets: List[PredictedDetection]
    ) -> List[int]:
        payload = {
            "model_name": model_name,
            "detections": [
                {
                    "boundary": _payload_for_bounding_polygon(det.boundary),
                    "scored_labels": [
                        asdict(scored_label)
                        for scored_label in det.scored_labels
                    ],
                    "image": asdict(det.image),
                }
                for det in dets
            ],
        }

        resp = self._requests_post_rel_host(
            "predicted-detections", json=payload
        )

        resp.raise_for_status()
        return resp.json()

    def upload_predicted_segmentations(
        self, model_name: str, segs: List[PredictedSegmentation]
    ) -> List[int]:
        payload = {
            "model_name": model_name,
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

        resp = self._requests_post_rel_host(
            "predicted-segmentations", json=payload
        )

        resp.raise_for_status()
        return resp.json()

    def upload_groundtruth_classifications(
        self, dataset_name: str, clfs: List[GroundTruthImageClassification]
    ) -> List[int]:
        payload = {
            "dataset_name": dataset_name,
            "classifications": [
                {
                    "labels": [asdict(label) for label in clf.labels],
                    "image": asdict(clf.image),
                }
                for clf in clfs
            ],
        }

        resp = self._requests_post_rel_host(
            "groundtruth-classifications", json=payload
        )
        resp.raise_for_status()

        return resp.json()

    def upload_predicted_classifications(
        self, model_name: str, clfs: List[PredictedImageClassification]
    ) -> List[int]:
        payload = {
            "model_name": model_name,
            "classifications": [
                {
                    "scored_labels": [
                        asdict(scored_label)
                        for scored_label in clf.scored_labels
                    ],
                    "image": asdict(clf.image),
                }
                for clf in clfs
            ],
        }

        resp = self._requests_post_rel_host(
            "predicted-classifications", json=payload
        )
        resp.raise_for_status()

        return resp.json()

    def create_dataset(self, name: str) -> "Dataset":
        self._requests_post_rel_host("datasets", json={"name": name})

        return Dataset(client=self, name=name)

    def delete_dataset(self, name: str) -> None:
        self._requests_delete_rel_host(f"datasets/{name}")

    def get_dataset(self, name: str) -> dict:
        resp = self._requests_get_rel_host("datasets", json={"name": name})

        return resp.json()

    def get_datasets(self) -> List[dict]:
        return self._requests_get_rel_host("datasets").json()

    def get_dataset_images(self, name: str) -> List[Image]:
        images = self._requests_get_rel_host(f"datasets/{name}/images").json()

        return [
            Image(
                uri=image["uri"], height=image["height"], width=image["width"]
            )
            for image in images
        ]

    def get_dataset_labels(self, name: str) -> List[Label]:
        labels = self._requests_get_rel_host(f"datasets/{name}/labels").json()

        return [
            Label(key=label["key"], value=label["value"]) for label in labels
        ]

    def create_model(self, name: str) -> "Model":
        self._requests_post_rel_host("models", json={"name": name})

        return Model(client=self, name=name)

    def delete_model(self, name: str) -> None:
        self._requests_delete_rel_host(f"models/{name}")

    def get_model(self, name: str) -> dict:
        resp = self._requests_get_rel_host("models", json={"name": name})

        return resp.json()

    def get_models(self) -> List[dict]:
        return self._requests_get_rel_host("models").json()

    def get_all_labels(self) -> List[Label]:
        return self._requests_get_rel_host("labels").json()

    def finalize_dataset(self, name: str):
        return self._requests_put_rel_host(f"datasets/{name}/finalize")


class Dataset:
    def __init__(self, client: Client, name: str):
        self.client = client
        self.name = name

    def add_groundtruth_detections(self, dets: List[GroundTruthDetection]):
        return self.client.upload_groundtruth_detections(
            dataset_name=self.name, dets=dets
        )

    def add_groundtruth_classifications(
        self, clfs: List[GroundTruthImageClassification]
    ):
        return self.client.upload_groundtruth_classifications(
            dataset_name=self.name, clfs=clfs
        )

    def add_groundtruth_segmentations(
        self, segs: List[GroundTruthSegmentation]
    ):
        return self.client.upload_groundtruth_segmentations(
            dataset_name=self.name, segs=segs
        )

    def finalize(self):
        return self.client.finalize_dataset(self.name)

    def delete(self):
        return self.client.delete_dataset(self.name)

    def get_images(self) -> List[Image]:
        return self.client.get_dataset_images(self.name)

    def get_labels(self) -> List[Label]:
        return self.client.get_dataset_labels(self.name)


class Model:
    def __init__(self, client: Client, name: str):
        self.client = client
        self.name = name

    def add_predicted_detections(self, dets: List[PredictedDetection]) -> None:
        return self.client.upload_predicted_detections(
            model_name=self.name, dets=dets
        )

    def add_predicted_segmentations(
        self, segs: List[PredictedSegmentation]
    ) -> None:
        return self.client.upload_predicted_segmentations(
            model_name=self.name, segs=segs
        )

    def add_predicted_classifications(
        self, clfs: List[PredictedImageClassification]
    ) -> None:
        return self.client.upload_predicted_classifications(
            model_name=self.name, clfs=clfs
        )
