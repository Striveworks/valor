import json
import math
import os
import time
import warnings
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import requests

from velour import schemas
from velour.enums import AnnotationType, JobStatus, TaskType


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

    def get_datasets(self) -> List[dict]:
        return self._requests_get_rel_host("datasets").json()

    def get_models(self) -> List[dict]:
        return self._requests_get_rel_host("models").json()

    def get_labels(self) -> List[schemas.Label]:
        return self._requests_get_rel_host("labels").json()

    def delete_dataset(self, name: str):
        try:
            self._requests_delete_rel_host(f"datasets/{name}")
        except ClientException as e:
            if "does not exist" not in str(e):
                raise e

    def delete_model(self, name: str):
        try:
            self._requests_delete_rel_host(f"models/{name}")
        except ClientException as e:
            if "does not exist" not in str(e):
                raise e


class Evaluation:
    def __init__(
        self,
        client: Client,
        dataset_name: str,
        model_name: str,
        job_id: int,
        **kwargs,
    ):
        self._id = job_id
        self.client = client
        self.dataset_name = dataset_name
        self.model_name = model_name

        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def status(self) -> str:
        resp = self.client._requests_get_rel_host(
            f"evaluations/{self._id}/dataset/{self.dataset_name}/model/{self.model_name}"
        ).json()
        return JobStatus(resp)

    # TODO: replace value with a dataclass?
    @property
    def settings(self) -> dict:
        return self.client._requests_get_rel_host(
            f"evaluations/{self._id}/dataset/{self.dataset_name}/model/{self.model_name}/settings"
        ).json()

    def wait_for_completion(self, *, interval=1.0, timeout=None):
        if timeout:
            timeout_counter = int(math.ceil(timeout / interval))
        while self.status not in [JobStatus.DONE, JobStatus.FAILED]:
            time.sleep(interval)
            if timeout:
                timeout_counter -= 1
                if timeout_counter < 0:
                    raise TimeoutError

    @property
    def metrics(self) -> List[dict]:
        if self.status != JobStatus.DONE:
            return []
        return self.client._requests_get_rel_host(
            f"evaluations/{self._id}/dataset/{self.dataset_name}/model/{self.model_name}/metrics"
        ).json()

    @property
    def confusion_matrices(self) -> List[dict]:
        if self.status != JobStatus.DONE:
            return []
        return self.client._requests_get_rel_host(
            f"evaluations/{self._id}/dataset/{self.dataset_name}/model/{self.model_name}/confusion-matrices"
        ).json()


class Dataset:
    def __init__(
        self,
        client: Client,
        info: schemas.Dataset,
    ):
        self.client = client
        self.info = info
        self._metadata = {
            metadatum.key: metadatum.value for metadatum in info.metadata
        }

    @property
    def id(self):
        return self.info.id

    @property
    def name(self):
        return self.info.name

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @classmethod
    def create(
        cls,
        client: Client,
        name: str,
        **kwargs,
    ):
        # Create the dataset on server side first to get ID info
        ds = schemas.Dataset(
            name=name,
            metadata=[],
        )
        for key in kwargs:
            ds.metadata.append(
                schemas.MetaDatum(
                    key=key,
                    value=kwargs[key],
                )
            )
        resp = client._requests_post_rel_host("datasets", json=asdict(ds))

        # @TODO: Handle this response
        if resp:
            pass

        # Retrive newly created dataset with its ID
        return cls.get(client, name)

    @classmethod
    def get(cls, client: Client, name: str):
        resp = client._requests_get_rel_host(f"datasets/{name}").json()
        metadata = [
            schemas.MetaDatum(
                key=metadatum["key"],
                value=metadatum["value"],
            )
            for metadatum in resp["metadata"]
        ]
        info = schemas.Dataset(
            name=resp["name"],
            id=resp["id"],
            metadata=metadata,
        )
        return cls(
            client=client,
            info=info,
        )

    def add_metadatum(self, metadatum: schemas.MetaDatum):
        # @TODO: Add endpoint to allow adding custom metadatums
        self.info.metadata.append(metadatum)
        self.__metadata__[metadatum.key] = metadatum

    def add_groundtruth(
        self,
        groundtruth: schemas.GroundTruth,
    ):
        try:
            assert isinstance(groundtruth, schemas.GroundTruth)
        except AssertionError:
            raise TypeError(f"Invalid type `{type(groundtruth)}`")

        if len(groundtruth.annotations) == 0:
            warnings.warn(
                f"GroundTruth for datum with uid `{groundtruth.datum.uid}` contains no annotations. Skipping..."
            )
            return

        groundtruth.datum.dataset = self.info.name
        self.client._requests_post_rel_host(
            "groundtruths",
            json=asdict(groundtruth),
        )

    def get_groundtruth(self, uid: str) -> schemas.GroundTruth:
        resp = self.client._requests_get_rel_host(
            f"groundtruths/dataset/{self.info.name}/datum/{uid}"
        ).json()
        return schemas.GroundTruth(**resp)

    def get_labels(self) -> List[schemas.LabelDistribution]:
        labels = self.client._requests_get_rel_host(
            f"labels/dataset/{self.name}"
        ).json()

        return [
            schemas.Label(key=label["key"], value=label["value"])
            for label in labels
        ]

    def get_datums(self) -> List[schemas.Datum]:
        """Returns a list of datums."""
        datums = self.client._requests_get_rel_host(
            f"data/dataset/{self.name}"
        ).json()
        return [schemas.Datum(**datum) for datum in datums]

    def get_images(self) -> List[schemas.ImageMetadata]:
        """Returns a list of Image Metadata if it exists, otherwise raises Dataset contains no images."""
        return [
            schemas.ImageMetadata.from_datum(datum)
            for datum in self.get_datums()
            if schemas.ImageMetadata.valid(datum)
        ]

    def get_evaluations(self) -> List[Evaluation]:
        model_evaluations = self.client._requests_get_rel_host(
            f"evaluations/datasets/{self.name}"
        ).json()
        return [
            Evaluation(
                client=self.client,
                dataset_name=self.name,
                model_name=model_name,
                job_id=job_id,
            )
            for model_name in model_evaluations
            for job_id in model_evaluations[model_name]
        ]

    def get_info(self) -> schemas.Info:
        resp = self.client._requests_get_rel_host(
            f"datasets/{self.name}/info"
        ).json()

        return schemas.Info(
            annotation_type=resp["annotation_type"],
            number_of_classifications=resp["number_of_classifications"],
            number_of_bounding_boxes=resp["number_of_bounding_boxes"],
            number_of_bounding_polygons=resp["number_of_bounding_polygons"],
            number_of_segmentations=resp["number_of_segmentation_rasters"],
            associated=resp["associated"],
        )

    def finalize(self):
        return self.client._requests_put_rel_host(
            f"datasets/{self.name}/finalize"
        )

    def delete(self):
        self.client._requests_delete_rel_host(f"datasets/{self.name}").json()
        del self


class Model:
    def __init__(
        self,
        client: Client,
        info: schemas.Model,
    ):
        self.client = client
        self.info = info
        self._metadata = {
            metadatum.key: metadatum.value for metadatum in info.metadata
        }

    @property
    def id(self):
        return self.info.id

    @property
    def name(self):
        return self.info.name

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @classmethod
    def create(
        cls,
        client: Client,
        name: str,
        **kwargs,
    ):
        # Create the dataset on server side first to get ID info
        md = schemas.Model(
            name=name,
            metadata=[],
        )
        for key in kwargs:
            md.metadata.append(
                schemas.MetaDatum(
                    key=key,
                    value=kwargs[key],
                )
            )
        resp = client._requests_post_rel_host("models", json=asdict(md))

        # @TODO: Handle this response
        if resp:
            pass

        # Retrive newly created dataset with its ID
        return cls.get(client, name)

    @classmethod
    def get(cls, client: Client, name: str):
        resp = client._requests_get_rel_host(f"models/{name}").json()
        metadata = [
            schemas.MetaDatum(
                key=metadatum["key"],
                value=metadatum["value"],
            )
            for metadatum in resp["metadata"]
        ]
        info = schemas.Model(
            name=resp["name"],
            id=resp["id"],
            metadata=metadata,
        )
        return cls(
            client=client,
            info=info,
        )

    def delete(self):
        self.client._requests_delete_rel_host(f"models/{self.name}").json()
        del self

    def add_metadatum(self, metadatum: schemas.MetaDatum):
        # @TODO: Add endpoint to allow adding custom metadatums
        self.info.metadata.append(metadatum)
        self.__metadata__[metadatum.key] = metadatum

    def add_prediction(self, prediction: schemas.Prediction):
        try:
            assert isinstance(prediction, schemas.Prediction)
        except AssertionError:
            raise TypeError(
                f"Expected `velour.schemas.Prediction`, got `{type(prediction)}`"
            )

        if len(prediction.annotations) == 0:
            warnings.warn(
                f"Prediction for datum with uid `{prediction.datum.uid}` contains no annotations. Skipping..."
            )
            return

        prediction.model = self.info.name
        return self.client._requests_post_rel_host(
            "predictions",
            json=asdict(prediction),
        )

    def get_prediction(self, datum: schemas.Datum) -> schemas.Prediction:
        resp = self.client._requests_get_rel_host(
            f"predictions/model/{self.info.name}/dataset/{datum.dataset}/datum/{datum.uid}",
        ).json()
        return schemas.Prediction(**resp)

    def finalize_inferences(self, dataset: "Dataset") -> None:
        return self.client._requests_put_rel_host(
            f"models/{self.name}/datasets/{dataset.name}/finalize"
        ).json()

    def evaluate_classification(
        self,
        dataset: Dataset,
        group_by: schemas.MetaDatum = schemas.MetaDatum(key="k", value="v"),
    ) -> Evaluation:
        """Start a classification evaluation job

        Parameters
        ----------
        dataset
            the dataset to evaluate against
        group_by
            optional name of metadatum to group the results by

        Returns
        -------
        Evaluation
            a job object that can be used to track the status of the job
            and get the metrics of it upon completion
        """
        payload = {
            "settings": {
                "model": self.name,
                "dataset": dataset.name,
            }
        }

        resp = self.client._requests_post_rel_host(
            "evaluations/clf-metrics", json=payload
        ).json()

        return Evaluation(
            client=self.client,
            dataset_name=dataset.name,
            model_name=self.name,
            **resp,
        )

    def evaluate_semantic_segmentation(self, dataset: Dataset) -> Evaluation:
        payload = {"settings": {"model": self.name, "dataset": dataset.name}}

        resp = self.client._requests_post_rel_host(
            "evaluations/semantic-segmentation-metrics", json=payload
        ).json()

        return Evaluation(
            client=self.client,
            dataset_name=dataset.name,
            model_name=self.name,
            **resp,
        )

    def evaluate_ap(
        self,
        dataset: "Dataset",
        task_type: TaskType = None,
        target_type: AnnotationType = None,
        iou_thresholds: List[float] = None,
        ious_to_keep: List[float] = None,
        min_area: float = None,
        max_area: float = None,
        label_key: Optional[str] = None,
    ) -> Evaluation:
        payload = {
            "settings": {
                "model": self.name,
                "dataset": dataset.name,
                "task_type": task_type,
                "target_type": target_type,
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
            "evaluations/ap-metrics", json=payload
        ).json()

        # resp should have keys "missing_pred_labels", "ignored_pred_labels", with values
        # list of label dicts. convert label dicts to Label objects

        for k in ["missing_pred_labels", "ignored_pred_labels"]:
            resp[k] = [schemas.Label(**la) for la in resp[k]]

        return Evaluation(
            client=self.client,
            dataset_name=dataset.name,
            model_name=self.name,
            **resp,
        )

    def get_evaluations(self) -> List[Evaluation]:
        dataset_evaluations = self.client._requests_get_rel_host(
            f"evaluations/models/{self.name}"
        ).json()
        return [
            Evaluation(
                client=self.client,
                dataset_name=dataset_name,
                model_name=self.name,
                job_id=job_id,
            )
            for dataset_name in dataset_evaluations
            for job_id in dataset_evaluations[dataset_name]
        ]

    def get_metric_dataframes(self):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Must have pandas installed to use `get_metric_dataframes`."
            )

        ret = []
        for evaluation in self.get_evaluations():
            metrics = [
                {**m, "dataset": evaluation.dataset_name}
                for m in evaluation.metrics
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
            ret.append({"settings": evaluation.settings, "df": df})

        return ret

    def get_labels(self) -> List[schemas.Label]:
        labels = self.client._requests_get_rel_host(
            f"labels/model/{self.name}"
        ).json()

        return [
            schemas.Label(key=label["key"], value=label["value"])
            for label in labels
        ]

    def get_label_distribution(self) -> Dict[schemas.Label, int]:
        distribution = self.client._requests_get_rel_host(
            f"models/{self.name}/labels/distribution"
        ).json()

        return {
            schemas.Label(
                key=label["label"]["key"], value=label["label"]["value"]
            ): {
                "count": label["count"],
                "scores": label["scores"],
            }
            for label in distribution
        }

    def get_info(self) -> schemas.Info:
        resp = self.client._requests_get_rel_host(
            f"models/{self.name}/info"
        ).json()

        return schemas.Info(
            annotation_type=resp["annotation_type"],
            number_of_classifications=resp["number_of_classifications"],
            number_of_bounding_boxes=resp["number_of_bounding_boxes"],
            number_of_bounding_polygons=resp["number_of_bounding_polygons"],
            number_of_segmentations=resp["number_of_segmentation_rasters"],
            associated=resp["associated"],
        )
