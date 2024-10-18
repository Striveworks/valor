import time
from collections import defaultdict

import evaluate
from nltk.tokenize import RegexpTokenizer
from nltk.translate import bleu_score
from valor_lite.text_generation import schemas
from valor_lite.text_generation.llm_clients import (
    InvalidLLMResponseError,
    LLMClient,
    MockLLMClient,
    WrappedMistralAIClient,
    WrappedOpenAIClient,
)
from valor_lite.text_generation.metric import (
    AnswerCorrectnessMetric,
    AnswerRelevanceMetric,
    BiasMetric,
    BLEUMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    ContextRelevanceMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    MetricType,
    ROUGEMetric,
    ROUGEType,
    SummaryCoherenceMetric,
    ToxicityMetric,
)

LLM_GUIDED_METRICS = {
    "AnswerCorrectness",
    "AnswerRelevance",
    "Bias",
    "ContextPrecision",
    "ContextRecall",
    "ContextRelevance",
    "Faithfulness",
    "Hallucination",
    "SummaryCoherence",
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
    llm_api_params: dict[str, str | int | dict],
) -> LLMClient:
    """
    Setup an LLM client for LLM guided evaluation.

    Parameters
    ----------
    llm_api_params : dict[str, str | int | dict], optional
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
    if "retries" in llm_api_params:
        client_kwargs["retries"] = llm_api_params["retries"]

    client = client_cls(**client_kwargs)
    client.connect()
    return client


def _compute_text_generation_metrics(
    data: list[tuple[str, list[str], int, str, list[str]]],
    metrics_to_return: list[MetricType] = [],
    llm_api_params: dict[str, str | int | dict] | None = None,
    metric_params: dict[str, dict] = {},
) -> list[
    AnswerCorrectnessMetric
    | AnswerRelevanceMetric
    | BiasMetric
    | BLEUMetric
    | ContextPrecisionMetric
    | ContextRecallMetric
    | ContextRelevanceMetric
    | FaithfulnessMetric
    | HallucinationMetric
    | ROUGEMetric
    | SummaryCoherenceMetric
    | ToxicityMetric
]:
    """
    Compute text generation metrics.

    Parameters
    ----------
    data: list[tuple[str, list[str], int, str, list[str]]]
        A list of tuples, where each tuple contains the prediction text, the prediction context list, the datum UID, the datum text, and the ground truth texts.
    metrics_to_return: list[MetricType]
        The list of metrics to compute, store, and return to the user.
    llm_api_params: dict[str, str | int | dict], optional
        A dictionary of parameters for the LLM API.
    metric_params: dict, optional
        A dictionary of optional parameters to pass in to specific metrics.

    Returns
    ----------
    Sequence[metrics.AnswerCorrectnessMetric | metrics.AnswerRelevanceMetric | metrics.BiasMetric | metrics.BLEUMetric | metrics.ContextPrecisionMetric | metrics.ContextRecallMetric | metrics.ContextRelevanceMetric | metrics.FaithfulnessMetric | metrics.HallucinationMetric | metrics.ROUGEMetric | metrics.SummaryCoherenceMetric | metrics.ToxicityMetric]
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
    is_SummaryCoherence_enabled = (
        MetricType.SummaryCoherence in metrics_to_return
    )
    is_Toxicity_enabled = MetricType.Toxicity in metrics_to_return

    client = None
    if any([metric in metrics_to_return for metric in LLM_GUIDED_METRICS]):
        if llm_api_params is None:
            raise ValueError(
                f"llm_api_params must be provided for the following metrics: {[metric for metric in metrics_to_return if metric in LLM_GUIDED_METRICS]}."
            )
        client = _setup_llm_client(llm_api_params)

    # Text comparison metrics require both predictions and ground truths.
    output = []
    if any(
        [metric in TEXT_COMPARISON_METRICS for metric in metrics_to_return]
    ):
        for (
            prediction_text,
            prediction_context_list,
            datum_uid,
            datum_text,
            groundtruth_texts,
        ) in data:
            if is_AnswerCorrectness_enabled:
                assert client
                try:
                    value = client.answer_correctness(
                        query=datum_text,
                        prediction=prediction_text,
                        groundtruth_list=groundtruth_texts,
                    )
                    output += [
                        AnswerCorrectnessMetric(
                            status="success",
                            value=value,
                            parameters={
                                "datum_uid": datum_uid,
                                "prediction": prediction_text,
                            },
                        )
                    ]
                except InvalidLLMResponseError:
                    output += [
                        AnswerCorrectnessMetric(
                            status="error",
                            value=None,
                            parameters={
                                "datum_uid": datum_uid,
                                "prediction": prediction_text,
                            },
                        )
                    ]

            if is_BLEU_enabled:
                bleu_params = metric_params.get("BLEU", {})
                if not isinstance(bleu_params, dict):
                    raise ValueError("BLEU parameters must be a dictionary.")
                weights = bleu_params.get("weights", [0.25, 0.25, 0.25, 0.25])
                bleu_metrics = _calculate_sentence_bleu(
                    predictions=prediction_text,
                    references=groundtruth_texts,
                    weights=weights,
                )

                output += [
                    BLEUMetric(
                        status="success",
                        value=metric["value"],
                        parameters={
                            "datum_uid": datum_uid,
                            "prediction": prediction_text,
                            "weights": weights,
                        },
                    )
                    for metric in bleu_metrics
                ]

            if is_ContextPrecision_enabled:
                assert client
                try:
                    value = client.context_precision(
                        query=datum_text,
                        ordered_context_list=prediction_context_list,
                        groundtruth_list=groundtruth_texts,
                    )
                    output += [
                        ContextPrecisionMetric(
                            status="success",
                            value=value,
                            parameters={
                                "datum_uid": datum_uid,
                                "context_list": prediction_context_list,
                            },
                        )
                    ]
                except InvalidLLMResponseError:
                    output += [
                        ContextPrecisionMetric(
                            status="error",
                            value=None,
                            parameters={
                                "datum_uid": datum_uid,
                                "context_list": prediction_context_list,
                            },
                        )
                    ]

            if is_ContextRecall_enabled:
                assert client
                try:
                    value = client.context_recall(
                        context_list=prediction_context_list,
                        groundtruth_list=groundtruth_texts,
                    )
                    output += [
                        ContextRecallMetric(
                            status="success",
                            value=value,
                            parameters={
                                "datum_uid": datum_uid,
                                "context_list": prediction_context_list,
                            },
                        )
                    ]
                except InvalidLLMResponseError:
                    output += [
                        ContextRecallMetric(
                            status="error",
                            value=None,
                            parameters={
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
                    predictions=prediction_text,
                    references=groundtruth_texts,
                    rouge_types=rouge_types,
                    use_stemmer=use_stemmer,
                )

                output += [
                    ROUGEMetric(
                        status="success",
                        value=metric["value"],
                        parameters={
                            "datum_uid": datum_uid,
                            "prediction": prediction_text,
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

        for (
            prediction_text,
            prediction_context_list,
            datum_uid,
            datum_text,
            _,
        ) in data:
            if is_AnswerRelevance_enabled:
                try:
                    value = client.answer_relevance(
                        query=datum_text,
                        text=prediction_text,
                    )
                    output += [
                        AnswerRelevanceMetric(
                            status="success",
                            value=value,
                            parameters={
                                "datum_uid": datum_uid,
                                "prediction": prediction_text,
                            },
                        )
                    ]
                except InvalidLLMResponseError:
                    output += [
                        AnswerRelevanceMetric(
                            status="error",
                            value=None,
                            parameters={
                                "datum_uid": datum_uid,
                                "prediction": prediction_text,
                            },
                        )
                    ]
            if is_Bias_enabled:
                try:
                    value = client.bias(text=prediction_text)
                    output += [
                        BiasMetric(
                            status="success",
                            value=value,
                            parameters={
                                "datum_uid": datum_uid,
                                "prediction": prediction_text,
                            },
                        )
                    ]
                except InvalidLLMResponseError:
                    output += [
                        BiasMetric(
                            status="error",
                            value=None,
                            parameters={
                                "datum_uid": datum_uid,
                                "prediction": prediction_text,
                            },
                        )
                    ]

            if is_ContextRelevance_enabled:
                try:
                    value = client.context_relevance(
                        query=datum_text, context_list=prediction_context_list
                    )
                    output += [
                        ContextRelevanceMetric(
                            status="success",
                            value=value,
                            parameters={
                                "datum_uid": datum_uid,
                                "context_list": prediction_context_list,
                            },
                        )
                    ]
                except InvalidLLMResponseError:
                    output += [
                        ContextRelevanceMetric(
                            status="error",
                            value=None,
                            parameters={
                                "datum_uid": datum_uid,
                                "context_list": prediction_context_list,
                            },
                        )
                    ]

            if is_Faithfulness_enabled:
                try:
                    value = client.faithfulness(
                        text=prediction_text,
                        context_list=prediction_context_list,
                    )
                    output += [
                        FaithfulnessMetric(
                            status="success",
                            value=value,
                            parameters={
                                "datum_uid": datum_uid,
                                "prediction": prediction_text,
                                "context_list": prediction_context_list,
                            },
                        )
                    ]
                except InvalidLLMResponseError:
                    output += [
                        FaithfulnessMetric(
                            status="error",
                            value=None,
                            parameters={
                                "datum_uid": datum_uid,
                                "prediction": prediction_text,
                                "context_list": prediction_context_list,
                            },
                        )
                    ]

            if is_Hallucination_enabled:
                try:
                    value = client.hallucination(
                        text=prediction_text,
                        context_list=prediction_context_list,
                    )
                    output += [
                        HallucinationMetric(
                            status="success",
                            value=value,
                            parameters={
                                "datum_uid": datum_uid,
                                "prediction": prediction_text,
                                "context_list": prediction_context_list,
                            },
                        )
                    ]
                except InvalidLLMResponseError:
                    output += [
                        HallucinationMetric(
                            status="error",
                            value=None,
                            parameters={
                                "datum_uid": datum_uid,
                                "prediction": prediction_text,
                                "context_list": prediction_context_list,
                            },
                        )
                    ]

            if is_SummaryCoherence_enabled:
                try:
                    value = client.summary_coherence(
                        text=datum_text,
                        summary=prediction_text,
                    )
                    output += [
                        SummaryCoherenceMetric(
                            status="success",
                            value=value,
                            parameters={
                                "datum_uid": datum_uid,
                                "prediction": prediction_text,
                            },
                        )
                    ]
                except InvalidLLMResponseError:
                    output += [
                        SummaryCoherenceMetric(
                            status="error",
                            value=None,
                            parameters={
                                "datum_uid": datum_uid,
                                "prediction": prediction_text,
                            },
                        )
                    ]

            if is_Toxicity_enabled:
                try:
                    value = client.toxicity(text=prediction_text)
                    output += [
                        ToxicityMetric(
                            status="success",
                            value=value,
                            parameters={
                                "datum_uid": datum_uid,
                                "prediction": prediction_text,
                            },
                        )
                    ]
                except InvalidLLMResponseError:
                    output += [
                        ToxicityMetric(
                            status="error",
                            value=None,
                            parameters={
                                "datum_uid": datum_uid,
                                "prediction": prediction_text,
                            },
                        )
                    ]

    return output


def validate_text_gen_metrics_to_return(
    metrics_to_return: list[MetricType],
) -> None:
    """
    Validate that the provided metrics are appropriate for text generation.

    Parameters
    ----------
    metrics_to_return : List[MetricType]
        A list of metrics that need to be validated against the task type.

    Raises
    ------
    ValueError
        If any of the provided metrics are not supported for text generation.
    """
    if not set(metrics_to_return).issubset(MetricType.text_generation()):
        raise ValueError(
            f"The following metrics are not supported for text generation: '{set(metrics_to_return) - MetricType.text_generation()}'"
        )


def validate_metric_parameters(
    metrics_to_return: list[MetricType],
    metric_params: dict[str, dict],
):
    # check that the keys of metric parameters are all in metrics_to_return
    if not set(metric_params.keys()).issubset(
        [metric.value for metric in metrics_to_return]
    ):
        raise ValueError(
            "The keys of metric_params must be a subset of the metrics_to_return."
        )

    if MetricType.BLEU in metric_params:
        bleu_params = metric_params[MetricType.BLEU.value]
        if "weights" in bleu_params:
            bleu_weights = bleu_params["weights"]
            if not all(
                isinstance(weight, (int, float)) and 0 <= weight
                for weight in bleu_weights
            ):
                raise ValueError(
                    "BLEU metric weights must be a list of non-negative integers or floats."
                )
            if sum(bleu_weights) != 1:
                raise ValueError("BLEU metric weights must sum to 1.")


def evaluate_text_generation(
    predictions: list[schemas.Prediction],
    metrics_to_return: list[MetricType],
    groundtruths: list[schemas.GroundTruth] = [],
    llm_api_params: dict[str, str | int | dict] | None = None,
    metric_params: dict[str, dict] = {},
) -> schemas.Evaluation:
    """
    Validates the parameters and formats the predictions and ground truths, then computes the text generation metrics.

    Parameters
    ----------
    predictions : list[schemas.Prediction]
        A list of predictions.
    metrics_to_return : list[MetricType]
        The list of metrics to compute, store, and return to the user. There is no default value, so the user must specify the metrics they want to compute.
    groundtruths : list[schemas.GroundTruth], optional
        A list of ground truths. Ground truths are not required for all text generation metrics.
    llm_api_params : dict[str, str | int | dict], optional
        A dictionary of parameters for the LLM API.
    metric_params : dict, optional
        A dictionary of optional parameters to pass in to specific metrics.

    Returns
    ----------
    schemas.Evaluation
        An evaluation object containing the computed metrics and metadata.
    """
    start_time = time.time()

    validate_text_gen_metrics_to_return(
        metrics_to_return=metrics_to_return,
    )
    validate_metric_parameters(
        metrics_to_return=metrics_to_return,
        metric_params=metric_params,
    )

    unique_datum_counts = len(set([p.datum.uid for p in predictions]))

    # Generate a list of data tuples, where each prediction is matched with its corresponding datum and the associated groundtruths.
    data = []
    for prediction in predictions:
        datum = prediction.datum
        for pred_annotation in prediction.annotations:
            groundtruth_annotations = []
            for gt in groundtruths:
                if gt.datum == datum:
                    groundtruth_annotations.extend(gt.annotations)
            groundtruth_texts = [
                gt_annotation.text for gt_annotation in groundtruth_annotations
            ]
            data.append(
                (
                    pred_annotation.text,
                    pred_annotation.context_list,
                    datum.uid,
                    datum.text,
                    groundtruth_texts,
                )
            )

    metrics = _compute_text_generation_metrics(
        data=data,
        metrics_to_return=metrics_to_return,
        llm_api_params=llm_api_params,
        metric_params=metric_params,
    )

    return schemas.Evaluation(
        parameters=schemas.EvaluationParameters(
            metrics_to_return=metrics_to_return,
            llm_api_params=llm_api_params,
            metric_params=metric_params,
        ),
        metrics=[metric.to_dict() for metric in metrics],
        meta={
            "datums": unique_datum_counts,
            "duration": time.time() - start_time,
        },
    )
