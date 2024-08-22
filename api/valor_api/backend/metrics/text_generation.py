from collections import defaultdict

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
    commit_results,
    log_evaluation_duration,
    log_evaluation_item_counts,
    prepare_filter_for_evaluation,
    validate_computation,
)
from valor_api.backend.query import generate_select
from valor_api.enums import MetricType, ROUGEType

LabelMapType = list[list[list[str]]]


LLM_GUIDED_METRICS = {
    "AnswerCorrectness",
    "AnswerRelevance",
    "Bias",
    "Coherence",
    "ContextPrecision",
    "ContextRecall",
    "ContextRelevance",
    "Faithfulness",
    "Hallucination",
    "Toxicity",
}


TEXT_COMPARISON_METRICS = {
    "AnswerCorrectness",
    "BLEU",
    "ContextPrecision",
    "ContextRecall",
    "ROUGE",
}


def _calculate_rouge_scores(
    predictions: str | list[str],
    references: list[str],
    rouge_types: list[ROUGEType] | None = None,
    use_stemmer: bool = False,
) -> list[dict[str, dict[str, float]]]:
    """
    Calculate ROUGE scores for a prediction (or list of predictions) given some set of references.

    Parameters
    ----------
    prediction: str | list[str]
        The prediction (or list of predictions) to score. Each prediction should be a string with tokens separated by spaces.
    references: list[str] | list[list[str]]
        A list of reference for a given prediction. Each reference should be a string with tokens separated by spaces.
    rouge_types: list[ROUGEType]
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
            "Received incorrect inputs. predictions should be a string and references a list of strings"
        )

    if rouge_types is None:
        rouge_types = [
            ROUGEType.ROUGE1,
            ROUGEType.ROUGE2,
            ROUGEType.ROUGEL,
            ROUGEType.ROUGELSUM,
        ]

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
        use_aggregator=False,  # aggregation gives us an average across all predictions, which isn't what we want
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
) -> list[dict[str, float]]:
    """
    Calculate sentence BLEU scores for a set of prediction - ground truth pairs.

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
            float(
                bleu_score.sentence_bleu(
                    references=tokenized_references,
                    hypothesis=tokenized_prediction,
                    weights=weights,
                ),  # type: ignore
            ),
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

    client_name = llm_api_params.get("client")
    if client_name is not None:
        match client_name:
            case "openai":
                client_cls = WrappedOpenAIClient
            case "mistral":
                client_cls = WrappedMistralAIClient
            case "mock":
                client_cls = MockLLMClient
            case _:
                raise ValueError(f"Client {client_name} is not supported.")
    else:
        raise NotImplementedError(
            "Support has not been implemented for api_url."
        )

    client_kwargs = {}
    if "api_key" in llm_api_params:
        client_kwargs["api_key"] = llm_api_params["api_key"]
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
    groundtruth_filter: schemas.Filter,
    prediction_filter: schemas.Filter,
    metrics_to_return: list[MetricType] = [],
    llm_api_params: dict[str, str | dict] | None = None,
    metric_params: dict = {},
) -> list[
    schemas.AnswerCorrectnessMetric
    | schemas.AnswerRelevanceMetric
    | schemas.BiasMetric
    | schemas.BLEUMetric
    | schemas.CoherenceMetric
    | schemas.ContextPrecisionMetric
    | schemas.ContextRecallMetric
    | schemas.ContextRelevanceMetric
    | schemas.FaithfulnessMetric
    | schemas.HallucinationMetric
    | schemas.ROUGEMetric
    | schemas.ToxicityMetric
]:
    """
    Compute text generation metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    datum_filter : schemas.Filter
        The filter to be used to query datums.
    groundtruth_filter : schemas.Filter
        The filter to be used to query ground truths.
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
    Sequence[schemas.AnswerCorrectnessMetric | schemas.AnswerRelevanceMetric | schemas.BiasMetric | schemas.BLEUMetric | schemas.CoherenceMetric | schemas.ContextPrecisionMetric | schemas.ContextRecallMetric | schemas.ContextRelevanceMetric | schemas.FaithfulnessMetric | schemas.HallucinationMetric | schemas.ROUGEMetric | schemas.ToxicityMetric]
        A list of computed metrics.
    """
    is_AnswerCorrectness_enabled = (
        MetricType.AnswerCorrectness in metrics_to_return
    )
    is_AnswerRelevance_enabled = (
        MetricType.AnswerRelevance in metrics_to_return
    )
    is_Bias_enabled = MetricType.Bias in metrics_to_return
    is_BLEU_enabled = MetricType.BLEU in metrics_to_return
    is_Coherence_enabled = MetricType.Coherence in metrics_to_return
    is_ContextPrecision_enabled = (
        MetricType.ContextPrecision in metrics_to_return
    )
    is_ContextRecall_enabled = MetricType.ContextRecall in metrics_to_return
    is_ContextRelevance_enabled = (
        MetricType.ContextRelevance in metrics_to_return
    )
    is_Faithfulness_enabled = MetricType.Faithfulness in metrics_to_return
    is_Hallucination_enabled = MetricType.Hallucination in metrics_to_return
    is_ROUGE_enabled = MetricType.ROUGE in metrics_to_return
    is_Toxicity_enabled = MetricType.Toxicity in metrics_to_return

    client = None
    if any([metric in metrics_to_return for metric in LLM_GUIDED_METRICS]):
        if llm_api_params is None:
            raise ValueError(
                f"llm_api_params must be provided for the following metrics: {[metric for metric in metrics_to_return if metric in LLM_GUIDED_METRICS]}."
            )
        client = _setup_llm_client(llm_api_params)

    prediction_subquery = (
        generate_select(
            models.Annotation.datum_id.label("datum_id"),
            models.Annotation.text.label("prediction_text"),
            models.Annotation.context_list.label("prediction_context_list"),
            label_source=models.Annotation,
            filters=prediction_filter,
        )
        .where(models.Annotation.model_id.isnot(None))
        .subquery()
    )

    # Text comparison metrics require both predictions and ground truths.
    output = []
    if any(
        [metric in TEXT_COMPARISON_METRICS for metric in metrics_to_return]
    ):
        # Use array_agg since there can be multiple ground truths and multiple predictions for a given datum_uid.
        groundtruth_subquery = (
            generate_select(
                models.Datum.id.label("datum_id"),
                models.Datum.uid.label("datum_uid"),
                models.Datum.text.label("datum_text"),
                models.Dataset.name.label("dataset_name"),
                functions.array_agg(models.Annotation.text).label(
                    "groundtruth_text"
                ),
                label_source=models.Annotation,
                filters=groundtruth_filter,
            )
            .where(models.Annotation.model_id.is_(None))
            .group_by(
                models.Datum.id.label("datum_id"),
                models.Datum.uid.label("datum_uid"),
                models.Datum.text.label("datum_text"),
                models.Dataset.name.label("dataset_name"),
            )
            .subquery()
        )

        joint_subquery = (
            select(
                groundtruth_subquery.c.datum_uid,
                groundtruth_subquery.c.dataset_name,
                groundtruth_subquery.c.datum_text,
                functions.array_agg(
                    prediction_subquery.c.prediction_text
                ).label("predictions"),
                functions.array_agg(
                    prediction_subquery.c.prediction_context_list
                ).label("list_of_prediction_context_lists"),
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
                groundtruth_subquery.c.datum_text,
            )
        )

        results = db.execute(joint_subquery).all()

        for (
            datum_uid,
            dataset_name,
            datum_text,
            predictions,
            list_of_prediction_context_lists,
            references,
        ) in results:
            if is_AnswerCorrectness_enabled:
                assert client
                for (prediction, groundtruth_list) in zip(
                    predictions, references
                ):
                    score = 0
                    for groundtruth in groundtruth_list:
                        score = max(
                            score,
                            client.answer_correctness(
                                query=datum_text,
                                prediction=prediction,
                                groundtruth=groundtruth,
                            ),
                        )
                    output += [
                        schemas.AnswerCorrectnessMetric(
                            value=score,
                            parameters={
                                "dataset": dataset_name,
                                "datum_uid": datum_uid,
                                "prediction": prediction,
                            },
                        )
                    ]

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

            if is_ContextPrecision_enabled:
                assert client
                for (prediction_context_list, groundtruth_list) in zip(
                    list_of_prediction_context_lists, references
                ):
                    score = 0
                    for groundtruth in groundtruth_list:
                        score = max(
                            score,
                            client.context_precision(
                                query=datum_text,
                                context_list=prediction_context_list,
                                groundtruth=groundtruth,
                            ),
                        )
                    output += [
                        schemas.ContextPrecisionMetric(
                            value=score,
                            parameters={
                                "dataset": dataset_name,
                                "datum_uid": datum_uid,
                                "context_list": prediction_context_list,
                            },
                        )
                    ]

            if is_ContextRecall_enabled:
                assert client
                for (prediction_context_list, groundtruth_list) in zip(
                    list_of_prediction_context_lists, references
                ):
                    score = 0
                    for groundtruth in groundtruth_list:
                        score = max(
                            score,
                            client.context_recall(
                                context_list=prediction_context_list,
                                groundtruth=groundtruth,
                            ),
                        )
                    output += [
                        schemas.ContextRecallMetric(
                            value=score,
                            parameters={
                                "dataset": dataset_name,
                                "datum_uid": datum_uid,
                                "context_list": prediction_context_list,
                            },
                        )
                    ]

            if is_ROUGE_enabled:
                rouge_params = metric_params.get("ROUGE", {})
                if not isinstance(rouge_params, dict):
                    raise ValueError("ROUGE parameters must be a dictionary.")
                rouge_types = rouge_params.get(
                    "rouge_types",
                    [
                        ROUGEType.ROUGE1,
                        ROUGEType.ROUGE2,
                        ROUGEType.ROUGEL,
                        ROUGEType.ROUGELSUM,
                    ],
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

    if any(
        [
            (
                metric_name in LLM_GUIDED_METRICS
                and metric_name not in TEXT_COMPARISON_METRICS
            )
            for metric_name in metrics_to_return
        ]
    ):
        assert client
        datum_subquery = (
            generate_select(
                models.Datum.id.label("datum_id"),
                models.Datum.uid.label("datum_uid"),
                models.Dataset.name.label("dataset_name"),
                models.Datum.text.label("datum_text"),
                label_source=models.Annotation,
                filters=datum_filter,
            )
            .distinct()
            .subquery()
        )

        joint_subquery = (
            select(
                datum_subquery.c.datum_uid.label("datum_uid"),
                datum_subquery.c.dataset_name.label("dataset_name"),
                datum_subquery.c.datum_text.label("datum_text"),
                prediction_subquery.c.prediction_text.label("prediction_text"),
                prediction_subquery.c.prediction_context_list.label(
                    "prediction_context_list"
                ),
            )
            .select_from(datum_subquery)
            .join(
                prediction_subquery,
                datum_subquery.c.datum_id == prediction_subquery.c.datum_id,
            )
        )

        results = db.execute(joint_subquery).all()

        for (
            datum_uid,
            dataset_name,
            datum_text,
            prediction_text,
            prediction_context_list,
        ) in results:
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
            if is_Bias_enabled:
                score = client.bias(text=prediction_text)
                output += [
                    schemas.BiasMetric(
                        value=score,
                        parameters={
                            "dataset": dataset_name,
                            "datum_uid": datum_uid,
                            "prediction": prediction_text,
                        },
                    )
                ]

            if is_Coherence_enabled:
                score = client.coherence(
                    text=datum_text,
                    summary=prediction_text,
                )
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

            if is_ContextRelevance_enabled:
                score = client.context_relevance(
                    query=datum_text, context_list=prediction_context_list
                )
                output += [
                    schemas.ContextRelevanceMetric(
                        value=score,
                        parameters={
                            "dataset": dataset_name,
                            "datum_uid": datum_uid,
                            "context_list": prediction_context_list,
                        },
                    )
                ]

            if is_Faithfulness_enabled:
                score = client.faithfulness(
                    text=prediction_text, context_list=prediction_context_list
                )
                output += [
                    schemas.FaithfulnessMetric(
                        value=score,
                        parameters={
                            "dataset": dataset_name,
                            "datum_uid": datum_uid,
                            "prediction": prediction_text,
                            "context_list": prediction_context_list,
                        },
                    )
                ]

            if is_Hallucination_enabled:
                score = client.hallucination(
                    text=prediction_text, context_list=prediction_context_list
                )
                output += [
                    schemas.HallucinationMetric(
                        value=score,
                        parameters={
                            "dataset": dataset_name,
                            "datum_uid": datum_uid,
                            "prediction": prediction_text,
                            "context_list": prediction_context_list,
                        },
                    )
                ]

            if is_Toxicity_enabled:
                score = client.toxicity(text=prediction_text)
                output += [
                    schemas.ToxicityMetric(
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
    parameters = schemas.EvaluationParameters(**evaluation.parameters)
    groundtruth_filter, prediction_filter = prepare_filter_for_evaluation(
        filters=schemas.Filter(**evaluation.filters),
        dataset_names=evaluation.dataset_names,
        model_name=evaluation.model_name,
        task_type=parameters.task_type,
    )

    assert (
        parameters.metrics_to_return
    ), "This will never be None. EvaluationParameters sets metrics_to_return during validation if it is None."

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
        datum_filter=groundtruth_filter,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        metrics_to_return=parameters.metrics_to_return,
        llm_api_params=parameters.llm_api_params,
        metric_params=metric_params,
    )

    # add metrics to database
    commit_results(db, metrics, evaluation_id)

    log_evaluation_duration(
        evaluation=evaluation,
        db=db,
    )

    return evaluation_id
