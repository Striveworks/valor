from dataclasses import dataclass, field

import pandas as pd  # TODO remove
from valor_lite.text_generation import annotation
from valor_lite.text_generation.computation import evaluate_text_generation
from valor_lite.text_generation.exceptions import (
    MismatchingTextGenerationDatumError,
)
from valor_lite.text_generation.metric import MetricType
from valor_lite.text_generation.utilities import validate_metrics_to_return


@dataclass
class ValorTextGenerationStreamingManager:
    """
    Manages the evaluation of text generation predictions streamed in one at a time or in batches.

    The streaming manager does not support ground truths, as ground truths are not available in real time.

    Attributes
    ----------
    metrics_to_return : list[MetricType]
        A list of metrics to calculate during the evaluation.
    llm_api_params : dict[str, str | int | dict], optional
        The parameters to setup the client with.
    metric_params: dict, optional
        A dictionary of optional parameters to pass in to specific metrics.
    joint_df : pd.DataFrame
        A DataFrame containing merged datum and prediction data.
    datum_uids : set
        A set of user specified unique identifiers for the data samples.
    """

    metrics_to_return: list[MetricType]
    llm_api_params: dict[str, str | int | dict]
    metric_params: dict[str, dict] = field(default_factory=dict)
    joint_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame([]))
    datum_uids: set = field(default_factory=set)
    _locked = False

    def __post_init__(self):
        """
        Validates parameters and locks the class attributes to prevent modification after initialization.

        Initializes the joint_df.
        """
        self._validate_streaming_metrics_to_return()
        self._initialize_joint_df()
        self._locked = True

    def __setattr__(self, key, value):
        """Overrides attribute setting to enforce immutability after initialization."""
        if (
            key
            in [
                "metrics_to_return",
                "llm_api_params",
                "metric_params",
            ]
        ) and self._locked:
            raise AttributeError(
                f"Cannot manually modify '{key}' after instantiation."
            )
        super().__setattr__(key, value)

    def _validate_streaming_metrics_to_return(self):
        """
        Validates that all metrics are text generation metrics and are not text comparison metrics.

        Ground truths will not be available in a streaming setting as the model makes predictions. If you were able to generate ground truths in real time, then you wouldn't need a model to generate predictions in the first place.
        """
        validate_metrics_to_return(
            metrics_to_return=self.metrics_to_return,
        )

        if set(self.metrics_to_return) & MetricType.text_comparison():
            raise ValueError(
                f"The following text generation metrics require groundtruths, so are not usable in a streaming setting: '{set(self.metrics_to_return) & MetricType.text_comparison()}'"
            )

    def _initialize_joint_df(self):
        """
        Initialize the joint_df if needed.

        Add a column to the joint_df for each metric.

        Validate the joint_df.
        """
        columns = [
            "datum_uid",
            "datum_text",
            "prediction_text",
            "prediction_context_list",
        ] + [metric._name_ for metric in self.metrics_to_return]

        # Initialize if no joint_df was specified
        if self.joint_df is None or self.joint_df.empty:
            self.joint_df = pd.DataFrame(
                [],
                columns=columns,
            )

        # Validation
        if not set(self.joint_df.columns) == set(columns):
            raise ValueError(
                "The joint_df columns do not match the expected columns. Please ensure that the joint_df is initialized correctly."
            )

        # Check that datum_uids are unique
        if not self.joint_df["datum_uid"].notnull().all():
            raise ValueError(
                "The joint_df contains rows with missing datum_uid values."
            )
        if not self.joint_df["datum_uid"].is_unique:
            raise ValueError(
                "The joint_df contains rows with non-unique datum_uid values."
            )

        # Check that for every row either prediction_text or prediction_context_list is not null. It's okay if one of the two is null, but not both.
        if (
            not self.joint_df[["prediction_text", "prediction_context_list"]]
            .notnull()
            .any(axis=1)
            .all()
        ):
            raise ValueError(
                "The joint_df contains rows that are missing both prediction_text and prediction_context_list. Every prediction in the joint_df must have either prediction_text or prediction_context_list."
            )

    def add_and_evaluate_prediction(
        self,
        predictions: list[annotation.Prediction],
    ) -> list[dict]:
        """
        Adds a prediction or batch of predictions and evaluates them.

        Parameters
        ----------
        predictions : list[annotation.Prediction]
            A list of Prediction objects.

        Returns
        -------
        list[dict]
            A list of computed metrics, returned as dictionaries.
        """
        if not (
            isinstance(predictions, list)
            and all(
                [isinstance(x, annotation.Prediction) for x in predictions]
            )
        ):
            raise TypeError(
                "predictions should be a list of annotation.Prediction objects."
            )
        if not len(predictions) > 0:
            raise ValueError(
                "No predictions were provided. Please provide at least one prediction."
            )

        for pred in predictions:
            # If self.joint_df has no rows, skip the next check
            if not self.joint_df.empty and pred.datum.uid in self.datum_uids:
                rows = self.joint_df[
                    self.joint_df["datum_uid"] == pred.datum.uid
                ]
                if not all(rows["datum_text"] == pred.datum.text):
                    raise MismatchingTextGenerationDatumError(
                        f"The provided prediction does not match the existing data for this datum_uid {pred.datum.uid}."
                    )

            for pred2 in predictions:
                if (
                    pred.datum.uid == pred2.datum.uid
                    and pred.datum.text != pred2.datum.text
                ):
                    raise MismatchingTextGenerationDatumError(
                        f"Two predictions with the same datum_uid {pred.datum.uid} have different datum text."
                    )

        metrics = evaluate_text_generation(
            predictions=predictions,
            metrics_to_return=self.metrics_to_return,
            llm_api_params=self.llm_api_params,
        )

        for pred in predictions:
            self.joint_df = pd.concat(
                [
                    self.joint_df,
                    pd.DataFrame(
                        [
                            {
                                "datum_uid": pred.datum.uid,
                                "datum_text": pred.datum.text,
                                "prediction_text": annotation.text,
                                "prediction_context_list": annotation.context_list,
                            }
                            for annotation in pred.annotations
                        ],
                    ),
                ],
                ignore_index=True,
            )

        # Conditions for matching computed metrics to the correct row.
        def conditions(
            row, datum_uid, prediction_text=None, prediction_context_list=None
        ):
            assert (
                prediction_text is not None
                or prediction_context_list is not None
            )

            if prediction_text is not None:
                if prediction_context_list is not None:
                    return (
                        row["datum_uid"] == datum_uid
                        and row["prediction_text"] == prediction_text
                        and row["prediction_context_list"]
                        == prediction_context_list
                    )
                else:
                    return (
                        row["datum_uid"] == datum_uid
                        and row["prediction_text"] == prediction_text
                    )
            else:
                return (
                    row["datum_uid"] == datum_uid
                    and row["prediction_context_list"]
                    == prediction_context_list
                )

        for m in metrics:
            metric_name = m["type"]
            value = m["value"]
            datum_uid = m["parameters"]["datum_uid"]
            prediction_text = m["parameters"].get("prediction", None)
            prediction_context_list = m["parameters"].get("context_list", None)

            # Set the metric value of the correct row.
            conditional_series = self.joint_df.apply(
                conditions,
                axis=1,
                args=(datum_uid, prediction_text, prediction_context_list),
            )
            self.joint_df.loc[conditional_series, metric_name] = value

        self.datum_uids.update([pred.datum.uid for pred in predictions])

        return metrics

    def get_results(
        self,
    ) -> pd.DataFrame:
        """
        Returns the joint_df with all predictions and computed metrics.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing all predictions and computed metrics.
        """
        return self.joint_df
