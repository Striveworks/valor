import io
import json
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
    ScoredLabel,
    _GroundTruthSegmentation,
    _PredictedSegmentation,
)
from velour.metrics import Task


class DatumTypes(Enum):
    IMAGE = "Image"
    TABULAR = "Tabular"

    @classmethod
    def invert(cls, value: str):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"the value {value} is not in enum {cls.__name__}.")


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
    def add_groundtruth(
        self, groundtruth: Union[List[List[Label]], Dict[str, List[Label]]]
    ):
        if isinstance(groundtruth, list):
            # make uids the list indices (as strings)
            groundtruth = {str(i): gt for i, gt in enumerate(groundtruth)}

        payload = {
            "dataset_name": self.name,
            "classifications": [
                {
                    "labels": [asdict(label) for label in labels],
                    "datum": {"uid": uid},
                }
                for uid, labels in groundtruth.items()
            ],
        }

        resp = self.client._requests_post_rel_host(
            "groundtruth-classifications", json=payload
        )

        return resp.json()


def _generate_chunks(
    name: str,
    data: list,
    chunk_size=100,
    progress_bar_title: str = "Chunking",
    show_progress_bar: bool = True,
):
    progress_bar = tqdm(
        total=len(data),
        unit="samples",
        unit_scale=True,
        desc=f"{progress_bar_title} ({name})",
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


class ImageDataset(DatasetBase):
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

        for chunk in _generate_chunks(
            self.name,
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


def _remove_none_from_dict(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


class ModelBase:
    def __init__(self, client: "Client", name: str):
        self.client = client
        self.name = name

    def finalize_inferences(self, dataset: DatasetBase) -> None:
        return self.client._requests_put_rel_host(
            f"models/{self.name}/inferences/{dataset.name}/finalize"
        ).json()

    def evaluate_classification(self, dataset: DatasetBase) -> "EvalJob":
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

    def get_evaluation_settings(self) -> List[dict]:
        # TODO: should probably have a dataclass for the output
        ret = self.client._requests_get_rel_host(
            f"models/{self.name}/evaluation-settings"
        ).json()

        return [_remove_none_from_dict(es) for es in ret]

    @staticmethod
    def _group_evaluation_settings(eval_settings: List[dict]):
        # return list of dicts with keys ids and (common) eval settings
        ret = []

        for es in eval_settings:
            es_without_id_dset_model = {
                k: v
                for k, v in es.items()
                if k not in ["id", "dataset_name", "model_name"]
            }
            found = False
            for grp in ret:
                if es_without_id_dset_model == grp["settings"]:
                    grp["ids"].append(es["id"])
                    grp["datasets"].append(es["dataset_name"])
                    found = True
                    break

            if not found:
                ret.append(
                    {
                        "ids": [es["id"]],
                        "settings": es_without_id_dset_model,
                        "datasets": [es["dataset_name"]],
                    }
                )

        return ret

    def get_metrics_at_evaluation_settings_id(
        self, eval_settings_id: int
    ) -> List[dict]:
        return [
            _remove_none_from_dict(m)
            for m in self.client._requests_get_rel_host(
                f"models/{self.name}/evaluation-settings/{eval_settings_id}/metrics"
            ).json()
        ]

    def get_confusion_matrices_at_evaluation_settings_id(
        self, eval_settings_id: int
    ) -> List[dict]:
        return self.client._requests_get_rel_host(
            f"/models/{self.name}/evaluation-settings/{eval_settings_id}/confusion-matrices"
        ).json()

    def get_metric_dataframes(self):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Must have pandas installed to use `get_metric_dataframes`."
            )
        eval_setting_groups = self._group_evaluation_settings(
            self.get_evaluation_settings()
        )

        ret = []
        for grp in eval_setting_groups:
            metrics = [
                {**m, "dataset": dataset}
                for id_, dataset in zip(grp["ids"], grp["datasets"])
                for m in self.get_metrics_at_evaluation_settings_id(id_)
            ]
            df = pd.DataFrame(metrics)
            for k in ["label", "parameters"]:
                df[k] = df[k].fillna("n/a")
            df["parameters"] = df["parameters"].apply(json.dumps)
            df["label"] = df["label"].apply(
                lambda x: f"{x['key']}: {x['value']}" if x != "n/a" else x
            )
            ret.append({"settings": grp["settings"], "df": df})

        return ret


class ImageModel(ModelBase):
    def add_predictions(
        self,
        dataset: ImageDataset,
        predictions: list,
        chunk_size: int = 1000,
        show_progress_bar: bool = True,
    ):
        log = []

        if not isinstance(predictions, list):
            raise ValueError("GroundTruth argument should be a list.")

        if len(predictions) == 0:
            raise ValueError("Empty list.")

        for chunk in _generate_chunks(
            self.name,
            predictions,
            chunk_size=chunk_size,
            progress_bar_title="Uploading",
            show_progress_bar=show_progress_bar,
        ):
            # Image Classification
            if isinstance(chunk[0], PredictedImageClassification):
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
                        for clf in predictions
                    ],
                }

                resp = self.client._requests_post_rel_host(
                    "predicted-classifications", json=payload
                )

                log += resp

            # Image Segmentation (Semantic, Instance)
            elif isinstance(chunk[0], _PredictedSegmentation):
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
                        for seg in predictions
                    ],
                }

                resp = self.client._requests_post_rel_host(
                    "predicted-segmentations", json=payload
                )

                log += resp

            # Object Detection
            elif isinstance(chunk[0], PredictedDetection):
                payload = {
                    "model_name": self.name,
                    "dataset_name": dataset.name,
                    "detections": [_det_to_dict(det) for det in predictions],
                }

                resp = self.client._requests_post_rel_host(
                    "predicted-detections", json=payload
                )

                log += resp

            # Unknown type.
            else:
                raise NotImplementedError(
                    f"Received predictions with type: '{type(chunk[0])}', which is not currently implemented."
                )

        return log

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


class TabularModel(ModelBase):
    def add_predictions(
        self,
        dataset: TabularDataset,
        predictions: Union[
            List[List[ScoredLabel]], Dict[str, List[ScoredLabel]]
        ],
    ):
        if isinstance(predictions, list):
            # make uids the list indices (as strings)
            predictions = {str(i): gt for i, gt in enumerate(predictions)}

        payload = {
            "model_name": self.name,
            "dataset_name": dataset.name,
            "classifications": [
                {
                    "scored_labels": [
                        asdict(scored_label) for scored_label in scored_labels
                    ],
                    "datum": {"uid": uid},
                }
                for uid, scored_labels in predictions.items()
            ],
        }

        resp = self.client._requests_post_rel_host(
            "predicted-classifications", json=payload
        )

        return resp.json()


class Client:
    """Client for interacting with the velour backend"""

    datum_type_and_entity_to_class = {
        "models": {
            DatumTypes.IMAGE: ImageModel,
            DatumTypes.TABULAR: TabularModel,
        },
        "datasets": {
            DatumTypes.IMAGE: ImageDataset,
            DatumTypes.TABULAR: TabularDataset,
        },
    }

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

    def _create_model_or_dataset(
        self,
        entity_type: str,
        datum_type: DatumTypes,
        name: str,
        href: str = None,
        description: str = None,
    ):
        entity_types = list(self.datum_type_and_entity_to_class.keys())
        if entity_type not in entity_types:
            raise ValueError(f"Expected `entity_type` to be in {entity_types}")

        class_ = self.datum_type_and_entity_to_class[entity_type][datum_type]

        self._requests_post_rel_host(
            entity_type,
            json={
                "name": name,
                "href": href,
                "description": description,
                "type": datum_type.value,
            },
        )

        return class_(client=self, name=name)

    def create_image_dataset(
        self, name: str, href: str = None, description: str = None
    ):
        return self._create_model_or_dataset(
            entity_type="datasets",
            datum_type=DatumTypes.IMAGE,
            name=name,
            href=href,
            description=description,
        )

    def create_tabular_dataset(
        self, name: str, href: str = None, description: str = None
    ):
        return self._create_model_or_dataset(
            entity_type="datasets",
            datum_type=DatumTypes.TABULAR,
            name=name,
            href=href,
            description=description,
        )

    def delete_dataset(self, name: str) -> None:
        job_id = self._requests_delete_rel_host(f"datasets/{name}").json()
        return Job(client=self, job_id=job_id)

    def _get_model_or_dataset(
        self,
        entity_type: str,
        name: str,
    ):
        entity_types = list(self.datum_type_and_entity_to_class.keys())
        if entity_type not in entity_types:
            raise ValueError(f"Expected `entity_type` to be in {entity_types}")

        resp = self._requests_get_rel_host(f"{entity_type}/{name}").json()

        datum_type = DatumTypes.invert(resp["type"])
        class_ = self.datum_type_and_entity_to_class[entity_type][datum_type]

        return class_(client=self, name=resp["name"])

    def get_model(self, name: str) -> ModelBase:
        return self._get_model_or_dataset(entity_type="models", name=name)

    def get_dataset(self, name: str) -> DatasetBase:
        return self._get_model_or_dataset(entity_type="datasets", name=name)

    def get_datasets(self) -> List[dict]:
        return self._requests_get_rel_host("datasets").json()

    def create_image_model(
        self, name: str, href: str = None, description: str = None
    ):
        return self._create_model_or_dataset(
            entity_type="models",
            datum_type=DatumTypes.IMAGE,
            name=name,
            href=href,
            description=description,
        )

    def create_tabular_model(
        self, name: str, href: str = None, description: str = None
    ):
        return self._create_model_or_dataset(
            entity_type="models",
            datum_type=DatumTypes.TABULAR,
            name=name,
            href=href,
            description=description,
        )

    def delete_model(self, name: str) -> None:
        self._requests_delete_rel_host(f"models/{name}")

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
