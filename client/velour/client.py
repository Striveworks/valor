import json
import math
import os

from dataclasses import asdict
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin

import requests
from tqdm.auto import tqdm

from velour import schemas, enums
from velour.metrics import Task


# def _payload_for_bounding_polygon(poly: BoundingPolygon) -> List[List[int]]:
#     """For converting a BoundingPolygon to list of list of ints expected
#     by the backend servce
#     """
#     return [[pt.x, pt.y] for pt in poly.points]


# def _list_of_list_to_bounding_polygon(points: List[List[int]]):
#     """Inverse of the above method"""
#     # backend should return a polygon with the same first and last entries
#     # but probably should change the backend to omit the last entry
#     # instead of doing it here
#     if points[0] != points[-1]:
#         raise ValueError("Expected points[0] == points[-1]")

#     return BoundingPolygon(points=[Point(*pt) for pt in points[:-1]])


# def _payload_for_polys_with_holes(
#     polys_with_holes: List[PolygonWithHole],
# ) -> List[Dict[str, List[List[int]]]]:
#     return [
#         {
#             "polygon": _payload_for_bounding_polygon(sh.polygon),
#             "hole": _payload_for_bounding_polygon(sh.hole)
#             if sh.hole is not None
#             else None,
#         }
#         for sh in polys_with_holes
#     ]


# def _det_to_dict(det: Union[PredictedDetection, GroundTruthDetection]) -> dict:
    # labels_key = (
    #     "scored_labels" if isinstance(det, PredictedDetection) else "labels"
    # )
    # ret = {
    #     labels_key: [asdict(label) for label in getattr(det, labels_key)],
    #     "image": asdict(det.image),
    # }
    # if det.is_bbox:
    #     ret["bbox"] = [
    #         det.bbox.xmin,
    #         det.bbox.ymin,
    #         det.bbox.xmax,
    #         det.bbox.ymax,
    #     ]
    # else:
    #     ret["boundary"] = _payload_for_bounding_polygon(det.boundary)

    # return ret


class ClientException(Exception):
    pass


class DatasetBase:
    def __init__(
        self,
        client: "Client",
        name: str,
        href: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.client = client
        self.name = name
        self.href = href
        self.description = description

    def add_groundtruth(
        self,
        groundtruth: schemas.GroundTruth,
        chunk_size: int = 1000,
        show_progress_bar: bool = True,
    ):

        if not isinstance(groundtruth, schemas.GroundTruth):
            raise ValueError
        
        payload = json.dumps(asdict(groundtruth))
        return self.client._requests_post_rel_host(
            "groundtruth", json=payload
        )
    
    def get_groundtruth(
        self, image_uid: str
    ) -> schemas.GroundTruth:
        resp = self.client._requests_get_rel_host(
            f"datasets/{self.name}/images/{image_uid}/groundtruth"
        ).json()
        return [schemas.GroundTruth(**gt) for gt in resp]

    def get_labels(self) -> List[schemas.LabelDistribution]:
        labels = self.client._requests_get_rel_host(
            f"datasets/{self.name}/labels"
        ).json()

        return [
            schemas.LabelDistribution(
                key=label["key"],
                value=label["value"],
                count=label["count"]
            )
            for label in labels
        ]
    
    def get_info(self) -> schemas.DatasetInfo:
        resp = self.client._requests_get_rel_host(
            f"datasets/{self.name}/info"
        ).json()

        return schemas.DatasetInfo(
            annotation_type=resp["annotation_type"],
            number_of_classifications=resp["number_of_classifications"],
            number_of_bounding_boxes=resp["number_of_bounding_boxes"],
            number_of_bounding_polygons=resp["number_of_bounding_polygons"],
            number_of_segmentations=resp["number_of_segmentation_rasters"],
            associated_datasets=None,
            associated_models=resp["associated"],
        )

    def finalize(self):
        return self.client._requests_put_rel_host(
            f"datasets/{self.name}/finalize"
        )

    def delete(self):
        return self.client.delete_dataset(self.name)


class ImageDataset(DatasetBase):
    def get_images(self) -> List[schemas.ImageMetadata]:
        images = self.client._requests_get_rel_host(
            f"datasets/{self.name}/images"
        ).json()

        return [
            schemas.ImageMetadata(
                uid=image["uid"], height=image["height"], width=image["width"]
            )
            for image in images
        ]


def _remove_none_from_dict(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


class ModelBase:
    def __init__(
        self,
        client: "Client",
        name: str,
        href: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.client = client
        self.name = name
        self.href = href
        self.description = description

    def finalize_inferences(self, dataset: DatasetBase) -> None:
        return self.client._requests_put_rel_host(
            f"models/{self.name}/inferences/{dataset.name}/finalize"
        ).json()

    def evaluate_classification(
        self, dataset: DatasetBase, group_by: str = None
    ) -> "EvalJob":
        """Start a classification evaluation job

        Parameters
        ----------
        dataset
            the dataset to evaluate against
        group_by
            optional name of metadatum to group the results by

        Returns
        -------
        EvalJob
            a job object that can be used to track the status of the job
            and get the metrics of it upon completion
        """
        payload = {
            "settings": {
                "model_name": self.name,
                "dataset_name": dataset.name,
                "group_by": group_by,
            }
        }

        resp = self.client._requests_post_rel_host(
            "clf-metrics", json=payload
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
            f"models/{self.name}/evaluation-settings/{eval_settings_id}/confusion-matrices"
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

            df = df.pivot(
                index=["type", "parameters", "label"], columns=["dataset"]
            )
            ret.append({"settings": grp["settings"], "df": df})

        return ret

    def get_labels(self) -> List[schemas.Label]:
        labels = self.client._requests_get_rel_host(
            f"models/{self.name}/labels"
        ).json()

        return [
            schemas.Label(key=label["key"], value=label["value"]) for label in labels
        ]

    def get_label_distribution(self) -> Dict[schemas.Label, int]:
        distribution = self.client._requests_get_rel_host(
            f"models/{self.name}/labels/distribution"
        ).json()

        return {
            schemas.Label(key=label["label"]["key"], value=label["label"]["value"]): {
                "count": label["count"],
                "scores": label["scores"],
            }
            for label in distribution
        }

    def get_info(self) -> schemas.ModelInfo:
        resp = self.client._requests_get_rel_host(
            f"models/{self.name}/info"
        ).json()

        return schemas.ModelInfo(
            annotation_type=resp["annotation_type"],
            number_of_classifications=resp["number_of_classifications"],
            number_of_bounding_boxes=resp["number_of_bounding_boxes"],
            number_of_bounding_polygons=resp["number_of_bounding_polygons"],
            number_of_segmentations=resp["number_of_segmentation_rasters"],
            associated_datasets=resp["associated"],
            associated_models=None,
        )


class ImageModel(ModelBase):
    def add_predictions(
        self,
        dataset: ImageDataset,
        predictions: list,
        chunk_size: int = 1000,
        show_progress_bar: bool = True,
    ):
        return []

    def evaluate_ap(
        self,
        dataset: ImageDataset,
        model_pred_task_type: Task = None,
        dataset_gt_task_type: Task = None,
        iou_thresholds: List[float] = None,
        ious_to_keep: List[float] = None,
        min_area: float = None,
        max_area: float = None,
        label_key: Optional[str] = None,
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
                "label_key": label_key,
            }
        }

        if iou_thresholds is not None:
            payload["iou_thresholds"] = iou_thresholds
        if ious_to_keep is not None:
            payload["ious_to_keep"] = ious_to_keep

        resp = self.client._requests_post_rel_host(
            "ap-metrics", json=payload
        ).json()
        # resp should have keys "missing_pred_labels", "ignored_pred_labels", with values
        # list of label dicts. convert label dicts to Label objects
        for k in ["missing_pred_labels", "ignored_pred_labels"]:
            resp[k] = [schemas.Label(**la) for la in resp[k]]

        return EvalJob(client=self.client, **resp)


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

        if endpoint[0] == "/":
            raise ValueError(
                "`endpoint` should not start with a forward slash."
            )

        url = urljoin(self.host, endpoint)
        requests_method = getattr(requests, method_name)

        if self.access_token is not None:
            headers = {"Authorization": f"Bearer {self.access_token}"}
        else:
            headers = None
        resp = requests_method(url, headers=headers, *args, **kwargs)
        if not resp.ok:
            try:
                raise ClientException(resp.json()["detail"])
            except (requests.exceptions.JSONDecodeError, KeyError):
                resp.raise_for_status()

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

    def create_dataset(
        self,
        name: str,
        metadata: List[schemas.Metadatum] = [],
    ):
        return self._requests_post_rel_host(
            "datasets",
            json={
                "info":{
                    "name": name, 
                    "id": None,
                    "metadata": metadata, 
                },
                "datums": [],
            }
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

        datum_type = enums.DataType.invert(resp["type"])
        class_ = self.datum_type_and_entity_to_class[entity_type][datum_type]

        return class_(
            client=self,
            name=resp["name"],
            href=resp["href"],
            description=resp["description"],
        )

    def get_model(self, name: str) -> ModelBase:
        return self._get_model_or_dataset(entity_type="models", name=name)

    def get_dataset(self, name: str) -> DatasetBase:
        return self._get_model_or_dataset(entity_type="datasets", name=name)

    def get_datasets(self) -> List[dict]:
        return self._requests_get_rel_host("datasets").json()

    def create_model(
        self, name: str, href: str = None, description: str = None
    ):
        return self._create_model_or_dataset(
            entity_type="models",
            datum_type=enums.DataType.IMAGE,
            name=name,
            href=href,
            description=description,
        )

    def delete_model(self, name: str) -> None:
        self._requests_delete_rel_host(f"models/{name}")

    def get_models(self) -> List[dict]:
        return self._requests_get_rel_host("models").json()

    def get_all_labels(self) -> List[schemas.Label]:
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
        resp = self.client._requests_get_rel_host(f"jobs/{self._id}").json()
        return resp["status"]


class EvalJob(Job):
    def metrics(self) -> List[dict]:
        return self.client._requests_get_rel_host(
            f"jobs/{self._id}/metrics"
        ).json()

    def confusion_matrices(self) -> List[dict]:
        return self.client._requests_get_rel_host(
            f"jobs/{self._id}/confusion-matrices"
        ).json()

    # TODO: replace value with a dataclass?
    def settings(self) -> dict:
        return self.client._requests_get_rel_host(
            f"jobs/{self._id}/settings"
        ).json()
