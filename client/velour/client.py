from dataclasses import asdict
from typing import List
from urllib.parse import urljoin

import requests
from velour.data_types import (
    BoundingPolygon,
    GroundTruthDetection,
    GroundTruthImageClassification,
    Image,
    Label,
    PredictedDetection,
    PredictedImageClassification,
)


def _payload_for_bounding_polygon(poly: BoundingPolygon) -> List[List[int]]:
    return [[pt.x, pt.y] for pt in poly.points]


class ClientException(Exception):
    pass


class Client:
    """Client for interacting with the velour backend"""

    def __init__(self, host: str):
        if not (host.startswith("http://") or host.startswith("https://")):
            raise ValueError(
                f"host must stat with 'http://' or 'https://' but got {host}"
            )
        self.host = host

    def _requests_wrapper(
        self, method_name: str, endpoint: str, *args, **kwargs
    ):
        assert method_name in ["get", "post", "put", "delete"]

        url = urljoin(self.host, endpoint)
        requests_method = getattr(requests, method_name)

        resp = requests_method(url, *args, **kwargs)
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

    def upload_predicted_detections(
        self, model_name: str, dets: List[PredictedDetection]
    ) -> List[int]:
        payload = {
            "model_name": model_name,
            "detections": [
                {
                    "boundary": _payload_for_bounding_polygon(det.boundary),
                    "labels": [asdict(label) for label in det.labels],
                    "image": asdict(det.image),
                    "score": det.score,
                }
                for det in dets
            ],
        }

        resp = self._requests_post_rel_host(
            "predicted-detections", json=payload
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
        self, dataset_name: str, clfs: List[PredictedImageClassification]
    ) -> List[int]:
        payload = {
            "dataset_name": dataset_name,
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

        return [Image(uri=image["uri"]) for image in images]

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

    def add_predictions(self, dets: List[PredictedDetection]) -> None:
        return self.client.upload_predicted_detections(
            model_name=self.name, dets=dets
        )
