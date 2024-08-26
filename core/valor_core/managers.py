import time
from dataclasses import dataclass, field

import pandas as pd
from valor_core import classification, detection, enums, schemas, utilities


@dataclass
class ValorDetectionManager:
    """
    Manages the evaluation of object detection predictions against groundtruths.

    Attributes
    ----------
    datum_uids : set[str]
        A set of unique identifiers for the data samples.
    label_map : dict[schemas.Label, schemas.Label]
        A mapping from one label schema to another.
    convert_annotations_to_type : AnnotationType, optional
        The target annotation type to convert the data to.
    metrics_to_return : list[enums.MetricType]
        A list of metrics to calculate during the evaluation.
    iou_thresholds_to_compute : list[float]
        A list of IoU thresholds to compute metrics for.
    iou_thresholds_to_return : list[float]
        A list of IoU thresholds to return metrics for.
    recall_score_threshold : float
        The score threshold for recall calculations.
    pr_curve_iou_threshold : float
        The IoU threshold used for precision-recall curve calculation.
    pr_curve_max_examples : int
        The maximum number of examples to include in the precision-recall curve.
    joint_df : pd.DataFrame
        A DataFrame containing merged groundtruth and prediction data with calculated IoU.
    detailed_joint_df : pd.DataFrame
        A DataFrame containing detailed data for precision-recall curves.
    unique_groundtruth_labels : dict[set[tuple[str, str]], set[str]]
        A dictionary mapping labels to unique groundtruth annotation IDs.
    unique_prediction_labels : set[tuple[str, str]]
        A set of unique labels present in the predictions.
    unique_annotation_ids : set[int]
        A set of unique annotation IDs across groundtruth and prediction data.
    """

    datum_uids: set = field(default_factory=set)
    label_map: dict[schemas.Label, schemas.Label] = field(default_factory=dict)
    convert_annotations_to_type: enums.AnnotationType | None = None
    metrics_to_return: list[enums.MetricType] = field(
        default_factory=lambda: [
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
        ]
    )
    iou_thresholds_to_compute: list[float] = field(
        default_factory=lambda: [round(0.5 + 0.05 * i, 2) for i in range(10)]
    )
    iou_thresholds_to_return: list[float] = field(
        default_factory=lambda: [0.5, 0.75]
    )
    recall_score_threshold: float = field(default=0.0)
    pr_curve_iou_threshold: float = field(default=0.5)
    pr_curve_max_examples: int = field(default=1)
    joint_df: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            [],
            columns=[
                "label_id",
                "id_gt",
                "label",
                "score",
                "id_pd",
                "iou_",
            ],
        )
    )
    detailed_joint_df: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            [],
            columns=[
                "datum_uid_gt",
                "label_key",
                "label_value_gt",
                "id_gt",
                "converted_geometry_gt",
                "datum_uid_pd",
                "label_value_pd",
                "score",
                "label_id_pd",
                "id_pd",
                "converted_geometry_pd",
                "is_label_match",
                "iou_",
            ],
        )
    )
    unique_groundtruth_labels: dict[set[tuple[str, str]], set[str]] = field(
        default_factory=dict
    )
    unique_prediction_labels: set[tuple[str, str]] = field(default_factory=set)
    unique_annotation_ids: set[int] = field(default_factory=set)
    _locked = False

    def __post_init__(self):
        """Validates parameters and locks the class attributes to prevent modification after initialization."""
        utilities.validate_label_map(self.label_map)
        utilities.validate_metrics_to_return(
            metrics_to_return=self.metrics_to_return,
            task_type=enums.TaskType.OBJECT_DETECTION,
        )
        utilities.validate_parameters(
            pr_curve_iou_threshold=self.pr_curve_iou_threshold,
            pr_curve_max_examples=self.pr_curve_max_examples,
            recall_score_threshold=self.recall_score_threshold,
        )
        self._locked = True

    def __setattr__(self, key, value):
        """Overrides attribute setting to enforce immutability after initialization."""

        if (
            key
            in [
                "label_map",
                "convert_annotations_to_type",
                "metrics_to_return",
                "iou_thresholds_to_compute",
                "iou_thresholds_to_return",
                "recall_score_threshold",
                "pr_curve_iou_threshold",
                "pr_curve_max_examples",
            ]
        ) and self._locked:
            raise AttributeError(
                f"Cannot manually modify '{key}' after instantiation."
            )
        super().__setattr__(key, value)

    def add_data(
        self,
        groundtruths: list[schemas.GroundTruth],
        predictions: list[schemas.Prediction],
    ) -> None:
        """
        Adds groundtruth and prediction data to the manager.

        Parameters
        ----------
        groundtruths : list[schemas.GroundTruth]
            A list of GroundTruth objects.
        predictions : list[schemas.Prediction]
            A list of Prediction objects.

        Raises
        ------
        ValueError
            If the groundtruths or predictions are not valid lists, or if duplicate
            datum_uids are detected.
        """
        if not (
            isinstance(groundtruths, list)
            and (len(groundtruths) > 0)
            and all([isinstance(x, schemas.GroundTruth) for x in groundtruths])
        ):
            raise ValueError(
                "groundtruths should be a non-empty list of schemas.GroundTruth objects."
            )

        if not (isinstance(predictions, list)):
            raise ValueError(
                "predictions should be a list of schemas.Prediction objects."
            )

        # check that datum_uids don't exist in the data yet
        unique_datum_uids = set([x.datum.uid for x in groundtruths]).union(
            set([x.datum.uid for x in predictions])
        )

        if not unique_datum_uids.isdisjoint(self.datum_uids):
            raise ValueError(
                "Attempted to add data for a datum_uid which already exists in this instantiated class."
            )

        (
            groundtruth_df,
            prediction_df,
            joint_df,
            detailed_joint_df,
        ) = detection.create_detection_evaluation_inputs(
            groundtruths=groundtruths,
            predictions=predictions,
            metrics_to_return=self.metrics_to_return,
            label_map=self.label_map,
            convert_annotations_to_type=self.convert_annotations_to_type,
        )

        # append these dataframes to self
        self.joint_df = utilities.concatenate_df_if_not_empty(
            df1=self.joint_df, df2=joint_df
        )
        self.detailed_joint_df = utilities.concatenate_df_if_not_empty(
            df1=self.detailed_joint_df, df2=detailed_joint_df
        )

        # add datums to self
        self.datum_uids = self.datum_uids.union(unique_datum_uids)

        # store unique labels (split by gt and pd) and unique annotations
        ids_per_label = (
            groundtruth_df.groupby(["label"])["id"].apply(set).to_dict()
        )

        for label, value in ids_per_label.items():
            if label in self.unique_groundtruth_labels.keys():
                self.unique_groundtruth_labels[
                    label
                ] = self.unique_groundtruth_labels[label].union(value)
            else:
                self.unique_groundtruth_labels[label] = value

        self.unique_prediction_labels.update(
            set(zip(prediction_df["label_key"], prediction_df["label_value"]))
        )
        self.unique_annotation_ids.update(
            set(groundtruth_df["annotation_id"])
            | set(prediction_df["annotation_id"])
        )

    def evaluate(self):
        """
        Evaluates the added data to compute detection metrics.

        Returns
        -------
        schemas.Evaluation
            An evaluation object containing metrics, confusion matrices, and metadata.

        Raises
        ------
        ValueError
            If the method is called before any data has been added.
        """
        if self.joint_df.empty:
            raise ValueError(
                "Attempted to call .evaluate() without adding any data first. Please use add_data to add data to this class."
            )

        start_time = time.time()

        # add the number of groundtruth observations per grouper to joint_df
        count_of_unique_ids_per_label = {
            key: len(value)
            for key, value in self.unique_groundtruth_labels.items()
        }

        self.joint_df["gts_per_grouper"] = self.joint_df["label"].map(
            count_of_unique_ids_per_label
        )

        metrics = detection.compute_detection_metrics(
            joint_df=self.joint_df,
            detailed_joint_df=self.detailed_joint_df,
            metrics_to_return=self.metrics_to_return,
            iou_thresholds_to_compute=self.iou_thresholds_to_compute,
            iou_thresholds_to_return=self.iou_thresholds_to_return,
            recall_score_threshold=self.recall_score_threshold,
            pr_curve_iou_threshold=self.pr_curve_iou_threshold,
            pr_curve_max_examples=self.pr_curve_max_examples,
        )

        missing_pred_labels = [
            (key, value)
            for key, value in (
                self.unique_groundtruth_labels.keys()
                - self.unique_prediction_labels
            )
        ]

        ignored_pred_labels = [
            (key, value)
            for key, value in (
                self.unique_prediction_labels
                - self.unique_groundtruth_labels.keys()
            )
        ]

        return schemas.Evaluation(
            parameters=schemas.EvaluationParameters(
                label_map=self.label_map,
                metrics_to_return=self.metrics_to_return,
                iou_thresholds_to_compute=self.iou_thresholds_to_compute,
                iou_thresholds_to_return=self.iou_thresholds_to_return,
                recall_score_threshold=self.recall_score_threshold,
                pr_curve_iou_threshold=self.pr_curve_iou_threshold,
                pr_curve_max_examples=self.pr_curve_max_examples,
            ),
            metrics=metrics,
            confusion_matrices=[],
            ignored_pred_labels=ignored_pred_labels,
            missing_pred_labels=missing_pred_labels,  # type: ignore - confirmed that this object is list[tuple[str, str]], but it isn't registerring as such
            meta={
                "labels": len(
                    self.unique_groundtruth_labels.keys()
                    | self.unique_prediction_labels
                ),
                "datums": len(self.datum_uids),
                "annotations": len(self.unique_annotation_ids),
                "duration": time.time() - start_time,
            },
        )


@dataclass
class ValorClassificationManager:
    """
    Manages the evaluation of classification predictions against groundtruths.

    Attributes
    ----------
    datum_uids : set[str]
        A set of unique identifiers for the data samples.
    label_map : dict[schemas.Label, schemas.Label]
        A mapping from one label schema to another.
    metrics_to_return : list[enums.MetricType]
        A list of metrics to calculate during the evaluation.
    pr_curve_max_examples : int
        The maximum number of examples to include in the precision-recall curve.
    joint_df : pd.DataFrame
        A DataFrame containing merged groundtruth and prediction data with calculated IoU.
    joint_df_filtered_on_best_score : pd.DataFrame
        A DataFrame containing merged groundtruth and prediction data with calculated IoU. Only joins on the best prediction for each groundtruth.
    unique_groundtruth_labels : dict[set[tuple[str, str]], set[str]]
        A dictionary mapping labels to unique groundtruth annotation IDs.
    unique_prediction_labels : set[tuple[str, str]]
        A set of unique labels present in the predictions.
    unique_annotation_ids : set[int]
        A set of unique annotation IDs across groundtruth and prediction data.
    """

    datum_uids: set = field(default_factory=set)
    label_map: dict[schemas.Label, schemas.Label] = field(default_factory=dict)
    metrics_to_return: list[enums.MetricType] = field(
        default_factory=lambda: [
            enums.MetricType.Precision,
            enums.MetricType.Recall,
            enums.MetricType.F1,
            enums.MetricType.Accuracy,
            enums.MetricType.ROCAUC,
        ]
    )
    pr_curve_max_examples: int = field(default=1)
    joint_df: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            [],
            [
                "datum_uid",
                "datum_id",
                "label_key",
                "label_value_gt",
                "id_gt",
                "annotation_id_gt",
                "label_value_pd",
                "score",
                "id_pd",
                "annotation_id_pd",
                "is_true_positive",
                "is_false_positive",
                "label",
            ],
        )
    )
    joint_df_filtered_on_best_score: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            [],
            columns=[
                "datum_id",
                "label_key",
                "label_value_gt",
                "label_value_pd",
                "best_score",
            ],
        )
    )
    unique_groundtruth_labels: dict[set[tuple[str, str]], set[str]] = field(
        default_factory=dict
    )
    unique_prediction_labels: set[tuple[str, str]] = field(default_factory=set)
    unique_annotation_ids: set[int] = field(default_factory=set)
    _locked = False

    def __post_init__(self):
        """Validates parameters and locks the class attributes to prevent modification after initialization."""
        utilities.validate_label_map(self.label_map)
        utilities.validate_metrics_to_return(
            metrics_to_return=self.metrics_to_return,
            task_type=enums.TaskType.CLASSIFICATION,
        )
        utilities.validate_parameters(
            pr_curve_max_examples=self.pr_curve_max_examples,
        )
        self._locked = True

    def __setattr__(self, key, value):
        """Overrides attribute setting to enforce immutability after initialization."""

        if (
            key
            in [
                "label_map",
                "metrics_to_return",
                "pr_curve_max_examples",
            ]
        ) and self._locked:
            raise AttributeError(
                f"Cannot manually modify '{key}' after instantiation."
            )
        super().__setattr__(key, value)

    def add_data(
        self,
        groundtruths: list[schemas.GroundTruth],
        predictions: list[schemas.Prediction],
    ) -> None:
        """
        Adds groundtruth and prediction data to the manager.

        Parameters
        ----------
        groundtruths : list[schemas.GroundTruth]
            A list of GroundTruth objects.
        predictions : list[schemas.Prediction]
            A list of Prediction objects.

        Raises
        ------
        ValueError
            If the groundtruths or predictions are not valid lists, or if duplicate
            datum_uids are detected.
        """
        if not (
            isinstance(groundtruths, list)
            and (len(groundtruths) > 0)
            and all([isinstance(x, schemas.GroundTruth) for x in groundtruths])
        ):
            raise ValueError(
                "groundtruths should be a non-empty list of schemas.GroundTruth objects."
            )

        if not (isinstance(predictions, list)):
            raise ValueError(
                "predictions should be a list of schemas.Prediction objects."
            )

        # check that datum_uids don't exist in the data yet
        unique_datum_uids = set([x.datum.uid for x in groundtruths]).union(
            set([x.datum.uid for x in predictions])
        )

        if not unique_datum_uids.isdisjoint(self.datum_uids):
            raise ValueError(
                "Attempted to add data for a datum_uid which already exists in this instantiated class."
            )

        (
            joint_df,
            joint_df_filtered_on_best_score,
        ) = classification.create_classification_evaluation_inputs(
            groundtruths=groundtruths,
            predictions=predictions,
            label_map=self.label_map,
        )

        # append these dataframes to self
        self.joint_df = utilities.concatenate_df_if_not_empty(
            df1=self.joint_df, df2=joint_df
        )
        self.joint_df_filtered_on_best_score = (
            utilities.concatenate_df_if_not_empty(
                df1=self.joint_df_filtered_on_best_score,
                df2=joint_df_filtered_on_best_score,
            )
        )

        # add datums to self
        self.datum_uids = self.datum_uids.union(unique_datum_uids)

        # store unique labels (split by gt and pd) and unique annotations
        ids_per_label = (
            joint_df[joint_df["label_value_gt"].notnull()]
            .groupby(["label"])["id_gt"]
            .apply(set)
            .to_dict()
        )

        for label, value in ids_per_label.items():
            if label in self.unique_groundtruth_labels.keys():
                self.unique_groundtruth_labels[
                    label
                ] = self.unique_groundtruth_labels[label].union(value)
            else:
                self.unique_groundtruth_labels[label] = value

        self.unique_prediction_labels.update(
            set(zip(joint_df["label_key"], joint_df["label_value_gt"]))
            | set(zip(joint_df["label_key"], joint_df["label_value_pd"]))
        )
        self.unique_annotation_ids.update(
            set(joint_df["annotation_id_gt"])
            | set(joint_df["annotation_id_pd"])
        )

    def evaluate(self):
        """
        Evaluates the added data to compute classification metrics.

        Returns
        -------
        schemas.Evaluation
            An evaluation object containing metrics, confusion matrices, and metadata.

        Raises
        ------
        ValueError
            If the method is called before any data has been added.
        """

        if self.joint_df.empty:
            raise ValueError(
                "Attempted to call .evaluate() without adding any data first. Please use add_data to add data to this class."
            )

        start_time = time.time()

        # add the number of groundtruth observations per grouper to joint_df
        count_of_unique_ids_per_label = {
            key: len(value)
            for key, value in self.unique_groundtruth_labels.items()
        }

        self.joint_df["gts_per_grouper"] = self.joint_df["label"].map(
            count_of_unique_ids_per_label
        )

        (
            confusion_matrices,
            metrics,
        ) = classification.compute_classification_metrics(
            joint_df=self.joint_df,
            joint_df_filtered_on_best_score=self.joint_df_filtered_on_best_score,
            metrics_to_return=self.metrics_to_return,
            pr_curve_max_examples=self.pr_curve_max_examples,
        )

        missing_pred_labels = [
            (key, value)
            for key, value in (
                self.unique_groundtruth_labels.keys()
                - self.unique_prediction_labels
            )
        ]

        ignored_pred_labels = [
            (key, value)
            for key, value in (
                self.unique_prediction_labels
                - self.unique_groundtruth_labels.keys()
            )
        ]

        return schemas.Evaluation(
            parameters=schemas.EvaluationParameters(
                label_map=self.label_map,
                metrics_to_return=self.metrics_to_return,
                pr_curve_max_examples=self.pr_curve_max_examples,
            ),
            metrics=metrics,
            confusion_matrices=confusion_matrices,
            ignored_pred_labels=ignored_pred_labels,
            missing_pred_labels=missing_pred_labels,  # type: ignore - confirmed that this object is list[tuple[str, str]], but it isn't registerring as such
            meta={
                "labels": len(
                    self.unique_groundtruth_labels.keys()
                    | self.unique_prediction_labels
                ),
                "datums": len(self.datum_uids),
                "annotations": len(self.unique_annotation_ids),
                "duration": time.time() - start_time,
            },
        )
