from collections import defaultdict
from typing import Sequence

import evaluate
from nltk.tokenize import RegexpTokenizer
from nltk.translate import bleu_score
from sqlalchemy.orm import Session
from sqlalchemy.sql import functions, select

from valor_api import schemas
from valor_api.backend import core, models
from valor_api.backend.core.llm_clients import (
    LLMClient,
    MockLLMClient,
    WrappedMistralAIClient,
    WrappedOpenAIClient,
)
from valor_api.backend.metrics.metric_utils import (
    create_metric_mappings,
    get_or_create_row,
    log_evaluation_duration,
    log_evaluation_item_counts,
    validate_computation,
)
from valor_api.backend.query import generate_query
from valor_api.enums import MetricType

LabelMapType = list[list[list[str]]]

LLM_GUIDED_METRICS = {
    "AnswerRelevance",
    "Coherence",
}

TEXT_COMPARISON_METRICS = {"BLEU", "ROUGE"}


def _calculate_rouge_scores(
    predictions: str | list[str],
    references: list[str],
    rouge_types: list[str] = ["rouge1", "rouge2", "rougeL", "rougeLsum"],
    use_stemmer: bool = False,
) -> list[dict]:
    """
    Calculate ROUGE scores for a prediction (or list of predictions) given some set of references.

    Parameters
    ----------
    prediction: str | list[str]
        The prediction (or list of predictions) to score. Each prediction should be a string with tokens separated by spaces.
    references: list[str] | list[list[str]]
        A list of reference for a given prediction. Each reference should be a string with tokens separated by spaces.
    rouge_types: list[str]
        A list of rouge types to calculate. Defaults to ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], where `rouge1` is unigram-based scoring, `rouge2` is bigram-based scoring, `rougeL` is scoring based on sentences (i.e., splitting on "." and ignoring "\n"), and `rougeLsum` is scoring based on splitting the text using "\n".
    use_stemmer: bool
        If True, uses Porter stemmer to strip word suffixes. Defaults to False.

    Raises
    ----------
    ValueError
        If prediction is neither a string nor a list.
    """
    if not predictions or not references or isinstance(references, str):
        raise ValueError(
            "Received incorrect inputs. predictions should be a string, references a list of strings, and weights a list/tuple of floats"
        )

    rouge = evaluate.load("rouge")

    # handle case where user passes in a single prediction
    if isinstance(predictions, str):
        processed_prediction = [predictions]
        processed_references = [references]
        # handle case where user passes multiple predictions
    elif isinstance(predictions, list) and all(
        [isinstance(lst, list) for lst in references]
    ):
        processed_prediction = predictions
        processed_references = references
    else:
        raise ValueError(
            "prediction should be a str or list[str]. If prediction is a list[str], then references must be a list of lists."
        )

    metrics = rouge.compute(
        predictions=processed_prediction,
        references=processed_references,
        rouge_types=rouge_types,
        use_stemmer=use_stemmer,
        use_aggregator=False,  # aggregation gives us an average across all predicitons, which isn't what we want
    )

    if not metrics:
        raise ValueError("No metrics were returned.")

    # find the max value for each prediction
    output = defaultdict(lambda: defaultdict(float))
    for i, prediction in enumerate(processed_prediction):
        for type_ in rouge_types:
            output[prediction][type_] = max(
                metrics[type_][i], output[prediction][type_]
            )

    return [
        {"prediction": prediction, "value": dict(value)}
        for prediction, value in output.items()
    ]


def _calculate_sentence_bleu(
    predictions: str | list[str],
    references: list[str] | list[list[str]],
    weights: list[float] = [0.25, 0.25, 0.25, 0.25],
) -> list[dict]:
    """
    Calculate sentence BLEU scores for a set of prediction-groundtruth pairs.

    Parameters
    ----------
    predictions: str | list[str]
        The predictions to score. Each prediction should be a string with tokens separated by spaces.
    references: list[str] | list[list[str]
        A list of reference for each prediction or a list of several references per prediction. Each reference should be a string with tokens separated by spaces.
    weights: list[float]
        The default BLEU calculates a score for up to 4-grams using uniform
        weights (this is called BLEU-4). To evaluate your translations with
        higher/lower order ngrams, use customized weights. Example: when accounting
        for up to 5-grams with uniform weights (this is called BLEU-5) use [1/5]*5
    """
    if (
        not predictions
        or not references
        or not weights
        or isinstance(references, str)
        or len(weights) == 0
    ):
        raise ValueError(
            "Received incorrect inputs. predictions should be a string, references a list of strings, and weights a list/tuple of floats"
        )

    # handle case where user passes in a single prediction
    if isinstance(predictions, str):
        processed_predictions = [predictions]
        processed_references = [references]
        # handle case where user passes multiple predictions
    elif isinstance(predictions, list) and all(
        [isinstance(lst, list) for lst in references]
    ):
        processed_predictions = predictions
        processed_references = references
    else:
        raise ValueError(
            "prediction should be a str or list[str]. If prediction is a list[str], then references must be a list of lists."
        )

    output = defaultdict(float)
    tokenizer = RegexpTokenizer(
        r"\w+|\$[\d]+|[^\s\.]+"
    )  # regex tokenizer that ignores periods

    for pred, refs in zip(processed_predictions, processed_references):

        tokenized_prediction = tokenizer.tokenize(pred)
        tokenized_references = [tokenizer.tokenize(ref) for ref in refs]

        # find the max value for each prediction
        output[pred] = max(
            bleu_score.sentence_bleu(
                references=tokenized_references,
                hypothesis=tokenized_prediction,
                weights=weights,
            ),  # type: ignore
            output[pred],
        )

    return [
        {"prediction": key, "value": value} for key, value in output.items()
    ]


def _setup_llm_client(
    llm_api_params: dict[str, str | dict],
) -> LLMClient:
    """
    Setup an LLM client for LLM guided evaluation.

    Parameters
    ----------
    llm_api_params : dict[str, str | dict], optional
        The parameters to setup the client with.

    Returns
    ----------
    LLMClient
        A wrapper for other LLM API clients.
    """
    if not ("client" in llm_api_params or "api_url" in llm_api_params):
        raise ValueError("Need to specify the client or api_url.")
    if "client" in llm_api_params and "api_url" in llm_api_params:
        raise ValueError("Cannot specify both client and api_url.")

    if llm_api_params.get("client") is not None:
        if llm_api_params["client"] == "openai":
            client_cls = WrappedOpenAIClient
        elif llm_api_params["client"] == "mistral":
            client_cls = WrappedMistralAIClient
        elif llm_api_params["client"] == "mock":
            client_cls = MockLLMClient
        else:
            raise ValueError(
                f"Client {llm_api_params['client']} is not supported."
            )
    else:
        raise NotImplementedError(
            "Support has not been implemented for api_url."
        )

    client_kwargs = {}
    if "data" in llm_api_params:
        if not isinstance(llm_api_params["data"], dict):
            raise ValueError("data must be a dictionary.")
        if "model" in llm_api_params["data"]:
            client_kwargs["model_name"] = llm_api_params["data"]["model"]
        if "seed" in llm_api_params["data"]:
            client_kwargs["seed"] = llm_api_params["data"]["seed"]

    client = client_cls(**client_kwargs)
    client.connect()
    return client


def _compute_text_generation_metrics(
    db: Session,
    datum_filter: schemas.Filter,
    prediction_filter: schemas.Filter,
    metrics_to_return: list[MetricType] = [],
    llm_api_params: dict[str, str | dict] | None = None,
    metric_params: dict = {},
) -> Sequence[
    schemas.AnswerRelevanceMetric
    | schemas.BLEUMetric
    | schemas.CoherenceMetric
    | schemas.ROUGEMetric
]:
    """
    Compute text generation metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    datum_filter : schemas.Filter
        The filter to be used to query datums.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    metrics_to_return: list[MetricType]
        The list of metrics to compute, store, and return to the user.
    llm_api_params: dict[str, str | dict], optional
        A dictionary of parameters for the LLM API.
    metric_params: dict, optional
        A dictionary of optional parameters to pass in to specific metrics.

    Returns
    ----------
    Sequence[schemas.AnswerRelevanceMetric | schemas.BLEUMetric | schemas.CoherenceMetric | schemas.ROUGEMetric]
        A list of computed metrics.
    """
    datum_subquery = generate_query(
        models.Datum,
        models.Datum.id.label("datum_id"),
        models.Datum.uid.label("datum_uid"),
        models.Dataset.name.label("dataset_name"),
        models.Datum.text.label("datum_text"),
        db=db,
        label_source=models.Annotation,
        filters=datum_filter,
    ).subquery()

    prediction_subquery = generate_query(
        models.Prediction,
        models.Annotation.datum_id.label("datum_id"),
        models.Annotation.text.label("prediction_text"),
        models.Annotation.context.label("prediction_context"),
        db=db,
        label_source=models.Prediction,
        filters=prediction_filter,
    ).subquery()

    output = []
    if any(
        [metric in TEXT_COMPARISON_METRICS for metric in metrics_to_return]
    ):
        # get reference text to compare against from groundtruths
        # use array_agg since there can be multiple references for a given datum_uid
        groundtruth_subquery = (
            generate_query(
                models.GroundTruth.id,
                models.Datum.id.label("datum_id"),
                models.Datum.uid.label("datum_uid"),
                models.Dataset.name.label("dataset_name"),
                functions.array_agg(models.Annotation.text).label(
                    "groundtruth_text"
                ),
                db=db,
                label_source=models.GroundTruth,
                filters=datum_filter,
            )
            .group_by(
                models.GroundTruth.id,
                models.Datum.id,
                models.Datum.uid,
                models.Dataset.name,
            )
            .subquery()
        )

        joint_subquery = (
            select(
                groundtruth_subquery.c.datum_uid,
                groundtruth_subquery.c.dataset_name,
                functions.array_agg(
                    prediction_subquery.c.prediction_text
                ).label("predictions"),
                functions.array_agg(
                    groundtruth_subquery.c.groundtruth_text
                ).label("references"),
            )
            .select_from(groundtruth_subquery)
            .join(
                prediction_subquery,
                groundtruth_subquery.c.datum_id
                == prediction_subquery.c.datum_id,
            )
            .group_by(
                groundtruth_subquery.c.datum_uid,
                groundtruth_subquery.c.dataset_name,
            )
        )

        results = db.execute(joint_subquery).all()
        is_BLEU_enabled = "BLEU" in metrics_to_return
        is_ROUGE_enabled = "ROUGE" in metrics_to_return

        for datum_uid, dataset_name, predictions, references in results:
            if is_BLEU_enabled:
                bleu_params = metric_params.get("BLEU", {})
                if not isinstance(bleu_params, dict):
                    raise ValueError("BLEU parameters must be a dictionary.")
                weights = bleu_params.get("weights", [0.25, 0.25, 0.25, 0.25])
                bleu_metrics = _calculate_sentence_bleu(
                    predictions=predictions,
                    references=references,
                    weights=weights,
                )

                output += [
                    schemas.BLEUMetric(
                        value=metric["value"],
                        parameters={
                            "dataset": dataset_name,
                            "datum_uid": datum_uid,
                            "prediction": metric["prediction"],
                            "weights": weights,
                        },
                    )
                    for metric in bleu_metrics
                ]
            if is_ROUGE_enabled:
                rouge_params = metric_params.get("ROUGE", {})
                if not isinstance(rouge_params, dict):
                    raise ValueError("ROUGE parameters must be a dictionary.")
                rouge_types = rouge_params.get(
                    "rouge_types", ["rouge1", "rouge2", "rougeL", "rougeLsum"]
                )
                use_stemmer = rouge_params.get("rouge_use_stemmer", False)
                rouge_metrics = _calculate_rouge_scores(
                    predictions=predictions,
                    references=references,
                    rouge_types=rouge_types,
                    use_stemmer=use_stemmer,
                )

                output += [
                    schemas.ROUGEMetric(
                        value=metric["value"],
                        parameters={
                            "dataset": dataset_name,
                            "datum_uid": datum_uid,
                            "prediction": metric["prediction"],
                            "rouge_types": rouge_types,
                            "use_stemmer": use_stemmer,
                        },
                    )
                    for metric in rouge_metrics
                ]

    client = None
    if any(
        [
            metric_name in LLM_GUIDED_METRICS
            for metric_name in metrics_to_return
        ]
    ):
        if llm_api_params is None:
            raise ValueError(
                f"llm_api_params must be provided for the following metrics: {[metric for metric in metrics_to_return if metric in LLM_GUIDED_METRICS]}."
            )
        client = _setup_llm_client(llm_api_params)

        total_query = (
            select(
                datum_subquery.c.datum_uid.label("datum_uid"),
                datum_subquery.c.dataset_name.label("dataset_name"),
                datum_subquery.c.datum_text.label("datum_text"),
                prediction_subquery.c.prediction_text.label("prediction_text"),
                prediction_subquery.c.prediction_context.label(
                    "prediction_context"
                ),
            )
            .select_from(datum_subquery)
            .join(
                prediction_subquery,
                datum_subquery.c.datum_id == prediction_subquery.c.datum_id,
            )
        )

        results = db.execute(total_query).all()
        is_AnswerRelevance_enabled = "AnswerRelevance" in metrics_to_return
        is_Coherence_enabled = "Coherence" in metrics_to_return

        for datum_uid, dataset_name, datum_text, prediction_text, _ in results:
            if is_AnswerRelevance_enabled:
                score = client.answer_relevance(
                    query=datum_text, text=prediction_text
                )
                output += [
                    schemas.AnswerRelevanceMetric(
                        value=score,
                        parameters={
                            "dataset": dataset_name,
                            "datum_uid": datum_uid,
                            "prediction": prediction_text,
                        },
                    )
                ]
            if is_Coherence_enabled:
                generated_text = prediction_text
                score = client.coherence(text=generated_text)
                output += [
                    schemas.CoherenceMetric(
                        value=score,
                        parameters={
                            "dataset": dataset_name,
                            "datum_uid": datum_uid,
                            "prediction": prediction_text,
                        },
                    )
                ]

    return output


@validate_computation
def compute_text_generation_metrics(
    *,
    db: Session,
    evaluation_id: int,
) -> int:
    """
    Compute text generation metrics. This function is intended to be run using FastAPI's `BackgroundTasks`.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    evaluation_id : int
        The job ID to create metrics for.

    Returns
    ----------
    int
        The evaluation job id.
    """

    # fetch evaluation
    evaluation = core.fetch_evaluation_from_id(db, evaluation_id)

    # unpack filters and params
    datum_filter = schemas.Filter(**evaluation.filters)
    prediction_filter = datum_filter.model_copy()
    groundtruth_filter = datum_filter.model_copy()
    parameters = schemas.EvaluationParameters(**evaluation.parameters)

    if parameters.metrics_to_return is None:
        raise ValueError("metrics_to_return must be provided.")

    log_evaluation_item_counts(
        db=db,
        evaluation=evaluation,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
    )

    metric_params = {}
    if parameters.bleu_weights is not None:
        if "BLEU" not in metric_params:
            metric_params["BLEU"] = {}
        metric_params["BLEU"]["weights"] = parameters.bleu_weights
    if parameters.rouge_types is not None:
        if "ROUGE" not in metric_params:
            metric_params["ROUGE"] = {}
        metric_params["ROUGE"]["rouge_types"] = parameters.rouge_types
    if parameters.rouge_use_stemmer is not None:
        if "ROUGE" not in metric_params:
            metric_params["ROUGE"] = {}
        metric_params["ROUGE"][
            "rouge_use_stemmer"
        ] = parameters.rouge_use_stemmer

    metrics = _compute_text_generation_metrics(
        db=db,
        datum_filter=datum_filter,
        prediction_filter=prediction_filter,
        metrics_to_return=parameters.metrics_to_return,
        llm_api_params=parameters.llm_api_params,
        metric_params=metric_params,
    )

    metric_mappings = create_metric_mappings(
        db=db,
        metrics=metrics,
        evaluation_id=evaluation.id,
    )

    for mapping in metric_mappings:
        # ignore value since the other columns are unique identifiers
        # and have empirically noticed value can slightly change due to floating
        # point errors
        get_or_create_row(
            db,
            models.Metric,
            mapping,
            columns_to_ignore=["value"],
        )

    log_evaluation_duration(
        evaluation=evaluation,
        db=db,
    )

    return evaluation_id