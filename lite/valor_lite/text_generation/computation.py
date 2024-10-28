from collections import defaultdict

import evaluate
from nltk.tokenize import RegexpTokenizer
from nltk.translate import bleu_score
from valor_lite.text_generation.exceptions import InvalidLLMResponseError
from valor_lite.text_generation.integrations import ClientWrapper
from valor_lite.text_generation.llm_instructions import (
    generate_answer_correctness_verdicts_instruction,
    generate_answer_relevance_verdicts_instruction,
    generate_bias_verdicts_instruction,
    generate_claims_instruction,
    generate_context_precision_verdicts_instruction,
    generate_context_recall_verdicts_instruction,
    generate_context_relevance_verdicts_instruction,
    generate_faithfulness_verdicts_instruction,
    generate_hallucination_verdicts_instruction,
    generate_opinions_instruction,
    generate_statements_instruction,
    generate_summary_coherence_instruction,
    generate_toxicity_verdicts_instruction,
)
from valor_lite.text_generation.utilities import trim_and_load_json


def generate_claims(
    client: ClientWrapper,
    system_prompt: str,
    text: str,
) -> list[str]:
    """
    Generate a list of claims from a piece of text, using a call to the LLM API.

    Parameters
    ----------
    text: str
        The text to extract claims from.

    Returns
    -------
    list[str]
        The list of claims extracted from the text.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": generate_claims_instruction(text=text),
        },
    ]

    response = client(messages)
    response = trim_and_load_json(response)
    if type(response) != dict or "claims" not in response:
        raise InvalidLLMResponseError(
            f"LLM response was not a dictionary or 'claims' was not in response: {response}"
        )
    claims = response["claims"]
    if type(claims) != list or not all(type(claim) == str for claim in claims):
        raise InvalidLLMResponseError(
            f"LLM response was not a valid list of claims (list[str]): {response}"
        )
    return claims


def generate_opinions(
    client: ClientWrapper,
    system_prompt: str,
    text: str,
) -> list[str]:
    """
    Generate a list of opinions from a piece of text, using a call to the LLM API.

    Parameters
    ----------
    text: str
        The text to extract opinions from.

    Returns
    -------
    list[str]
        The list of opinions extracted from the text.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": generate_opinions_instruction(text=text),
        },
    ]

    response = client(messages)
    response = trim_and_load_json(response)
    if type(response) != dict or "opinions" not in response:
        raise InvalidLLMResponseError(
            f"LLM response was not a dictionary or 'opinions' was not in response: {response}"
        )
    opinions = response["opinions"]
    if type(opinions) != list or not all(
        type(opinion) == str for opinion in opinions
    ):
        raise InvalidLLMResponseError(
            f"LLM response was not a valid list of opinions (list[str]): {response}"
        )
    return opinions


def generate_statements(
    client: ClientWrapper,
    system_prompt: str,
    text: str,
) -> list[str]:
    """
    Generate a list of statements from a piece of text, using a call to the LLM API.

    Parameters
    ----------
    text: str
        The text to extract statements from.

    Returns
    -------
    list[str]
        The list of statements extracted from the text.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": generate_statements_instruction(text=text),
        },
    ]

    response = client(messages)
    response = trim_and_load_json(response)
    if type(response) != dict or "statements" not in response:
        raise InvalidLLMResponseError(
            f"LLM response was not a dictionary or 'statements' was not in response: {response}"
        )
    statements = response["statements"]
    if type(statements) != list or not all(
        type(statement) == str for statement in statements
    ):
        raise InvalidLLMResponseError(
            f"LLM response was not a valid list of statements (list[str]): {response}"
        )
    return statements


def generate_answer_correctness_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    query: str,
    prediction_statements: list[str],
    groundtruth_statements: list[str],
) -> dict[str, list[dict[str, str]]]:
    """
    Generate lists of true positives, false positives and false negatives, using a call to the LLM API.

    Parameters
    ----------
    query: str
        The query that both the prediction and ground truth should be answering.
    prediction_statements: list[str]
        The prediction statements to evaluate.
    groundtruth_statements: list[str]
        The ground truth statements to evaluate.

    Returns
    -------
    dict[str, list[dict[str, str]]]
        A dictionary of true positives, false positives and false negatives.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": generate_answer_correctness_verdicts_instruction(
                query=query,
                prediction_statements=prediction_statements,
                groundtruth_statements=groundtruth_statements,
            ),
        },
    ]
    response = client(messages)
    response = trim_and_load_json(response)
    if (
        type(response) != dict
        or "TP" not in response
        or "FP" not in response
        or "FN" not in response
    ):
        raise InvalidLLMResponseError(
            f"LLM response was not a dictionary of true positives, false positives and false negatives: {response}"
        )

    if (
        type(response["TP"]) != list
        or type(response["FP"]) != list
        or type(response["FN"]) != list
    ):
        raise InvalidLLMResponseError(
            f"LLM response did not contain valid lists of true positives, false positives and false negatives: {response}"
        )

    if len(response["TP"]) + len(response["FP"]) != len(prediction_statements):
        raise InvalidLLMResponseError(
            f"Number of true positives and false positives did not match the number of prediction statements: {response}"
        )

    if len(response["FN"]) > len(groundtruth_statements):
        raise InvalidLLMResponseError(
            f"Number of false negatives exceeded the number of ground truth statements: {response}"
        )

    return response


def generate_answer_relevance_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    query: str,
    statements: list[str],
) -> list[dict[str, str]]:
    """
    Generate a list of answer relevance verdicts for a list of statements, using a call to the LLM API.

    Parameters
    ----------
    query: str
        The query to evaluate the statements against.
    statements: list[str]
        The statements to evaluate the validity of.

    Returns
    -------
    list[dict[str,str]]
        The list of verdicts for each statement. Each verdict is a dictionary with the "verdict" field.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": generate_answer_relevance_verdicts_instruction(
                query=query,
                statements=statements,
            ),
        },
    ]

    response = client(messages)
    response = trim_and_load_json(response)
    if type(response) != dict or "verdicts" not in response:
        raise InvalidLLMResponseError(
            f"LLM response was not a list of valid verdicts: {response}"
        )

    verdicts = response["verdicts"]
    if (
        type(verdicts) != list
        or len(verdicts) != len(statements)
        or not all(
            verdict["verdict"] in ["yes", "no", "idk"] for verdict in verdicts
        )
    ):
        raise InvalidLLMResponseError(
            f"LLM response was not a list of valid verdicts: {response}"
        )

    return verdicts


def generate_bias_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    opinions: list[str],
) -> list[dict[str, str]]:
    """
    Generate a list of bias verdicts for a list of opinions, using a call to the LLM API.

    Parameters
    ----------
    opinions: list[str]
        The opinions to evaluate the bias of.

    Returns
    -------
    list[dict[str,str]]
        The list of verdicts for each opinion. Each verdict is a dictionary with the "verdict" field.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": generate_bias_verdicts_instruction(
                opinions=opinions,
            ),
        },
    ]

    response = client(messages)
    response = trim_and_load_json(response)
    if type(response) != dict or "verdicts" not in response:
        raise InvalidLLMResponseError(
            f"LLM response was not a list of valid verdicts: {response}"
        )

    verdicts = response["verdicts"]
    if (
        type(verdicts) != list
        or len(verdicts) != len(opinions)
        or not all(verdict["verdict"] in ["yes", "no"] for verdict in verdicts)
    ):
        raise InvalidLLMResponseError(
            f"LLM response was not a list of valid verdicts: {response}"
        )

    return verdicts


def generate_context_precision_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    query: str,
    ordered_context_list: list[str],
    groundtruth: str,
) -> list[dict[str, str]]:
    """
    Generate a list of context precision verdicts for an ordered list of contexts, using a call to the LLM API.

    The verdict for each context should be 'yes' if the context is relevant to produce the ground truth answer to the query. The verdict should be 'no' otherwise.

    Parameters
    ----------
    query: str
        The query.
    ordered_context_list: list[str]
        The ordered list of contexts. Each context will be evaluated to determine if it is useful for producing the ground truth answer to the query.
    groundtruth: str
        The ground truth answer to the query.

    Returns
    -------
    list[dict[str,str]]
        The list of verdicts for each context. Each verdict is a dictionary with the "verdict" field.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": generate_context_precision_verdicts_instruction(
                query=query,
                ordered_context_list=ordered_context_list,
                groundtruth=groundtruth,
            ),
        },
    ]

    response = client(messages)
    response = trim_and_load_json(response)
    if type(response) != dict or "verdicts" not in response:
        raise InvalidLLMResponseError(
            f"LLM response was not a list of valid verdicts: {response}"
        )

    verdicts = response["verdicts"]
    if (
        type(verdicts) != list
        or len(verdicts) != len(ordered_context_list)
        or not all(verdict["verdict"] in ["yes", "no"] for verdict in verdicts)
    ):
        raise InvalidLLMResponseError(
            f"LLM response was not a list of valid verdicts: {response}"
        )

    return verdicts


def generate_context_recall_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    context_list: list[str],
    groundtruth_statements: list[str],
) -> list[dict[str, str]]:
    """
    Generate a list of context recall verdicts for a list of ground truth statements, using a call to the LLM API.

    The verdict for each ground truth statement should be 'yes' if the ground truth statement is attributable to the context list and 'no' otherwise.

    Parameters
    ----------
    context_list: list[str]
        The list of contexts to evaluate against.
    groundtruth_statements: str
        A list of statements extracted from the ground truth answer.

    Returns
    -------
    list[dict[str,str]]
        The list of verdicts for each ground truth statement. Each verdict is a dictionary with the "verdict" field.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": generate_context_recall_verdicts_instruction(
                context_list=context_list,
                groundtruth_statements=groundtruth_statements,
            ),
        },
    ]

    response = client(messages)
    response = trim_and_load_json(response)
    if type(response) != dict or "verdicts" not in response:
        raise InvalidLLMResponseError(
            f"LLM response was not a list of valid verdicts: {response}"
        )

    verdicts = response["verdicts"]
    if (
        type(verdicts) != list
        or len(verdicts) != len(groundtruth_statements)
        or not all(verdict["verdict"] in ["yes", "no"] for verdict in verdicts)
    ):
        raise InvalidLLMResponseError(
            f"LLM response was not a list of valid verdicts: {response}"
        )

    return verdicts


def generate_context_relevance_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    query: str,
    context_list: list[str],
) -> list[dict[str, str]]:
    """
    Generate a list of context relevance verdicts for a list of contexts, using a call to the LLM API.

    Parameters
    ----------
    query: str
        The query to evaluate each context against.
    context_list: list[str]
        The ordered list of contexts to evaluate the relevance of.

    Returns
    -------
    list[dict[str,str]]
        The list of verdicts for each context. Each verdict is a dictionary with the "verdict" field.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": generate_context_relevance_verdicts_instruction(
                query=query,
                context_list=context_list,
            ),
        },
    ]

    response = client(messages)
    response = trim_and_load_json(response)
    if type(response) != dict or "verdicts" not in response:
        raise InvalidLLMResponseError(
            f"LLM response was not a list of valid verdicts: {response}"
        )

    verdicts = response["verdicts"]
    if (
        type(verdicts) != list
        or len(verdicts) != len(context_list)
        or not all(verdict["verdict"] in ["yes", "no"] for verdict in verdicts)
    ):
        raise InvalidLLMResponseError(
            f"LLM response was not a list of valid verdicts: {response}"
        )

    return verdicts


def generate_faithfulness_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    claims: list[str],
    context_list: list[str],
) -> list[dict[str, str]]:
    """
    Generate a list of faithfulness verdicts for a list of claims, using a call to the LLM API.

    Parameters
    ----------
    claims: list[str]
        The claims to evaluate the faithfulness of.
    context_list: list[str]
        The list of contexts to evaluate against.

    Returns
    -------
    list[dict[str,str]]
        The list of verdicts for each claim. Each verdict is a dictionary with one key "verdict".
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": generate_faithfulness_verdicts_instruction(
                claims=claims,
                context_list=context_list,
            ),
        },
    ]

    response = client(messages)
    response = trim_and_load_json(response)
    if type(response) != dict or "verdicts" not in response:
        raise InvalidLLMResponseError(
            f"LLM response was not a list of valid verdicts: {response}"
        )

    verdicts = response["verdicts"]
    if (
        type(verdicts) != list
        or len(verdicts) != len(claims)
        or not all(verdict["verdict"] in ["yes", "no"] for verdict in verdicts)
    ):
        raise InvalidLLMResponseError(
            f"LLM response was not a list of valid verdicts: {response}"
        )

    return verdicts


def generate_hallucination_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    text: str,
    context_list: list[str],
) -> list[dict[str, str]]:
    """
    Generate a list of hallucination verdicts for a list of contexts, using a call to the LLM API.

    The verdict for each context should be 'yes' if the text contradicts that context. The verdict should be 'no' otherwise.

    Parameters
    ----------
    text: str
        The text to evaluate for hallucination.
    context_list: list[str]
        The list of contexts to compare against.

    Returns
    -------
    list[dict[str,str]]
        The list of verdicts for each context. Each verdict is a dictionary with the "verdict" field.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": generate_hallucination_verdicts_instruction(
                text=text,
                context_list=context_list,
            ),
        },
    ]

    response = client(messages)
    response = trim_and_load_json(response)
    if type(response) != dict or "verdicts" not in response:
        raise InvalidLLMResponseError(
            f"LLM response was not a list of valid verdicts: {response}"
        )

    verdicts = response["verdicts"]
    if (
        type(verdicts) != list
        or len(verdicts) != len(context_list)
        or not all(verdict["verdict"] in ["yes", "no"] for verdict in verdicts)
    ):
        raise InvalidLLMResponseError(
            f"LLM response was not a list of valid verdicts: {response}"
        )

    return verdicts


def generate_toxicity_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    opinions: list[str],
) -> list[dict[str, str]]:
    """
    Generate a list of toxicity verdicts for a list of opinions, using a call to the LLM API.

    Parameters
    ----------
    opinions: list[str]
        The opinions to evaluate the toxicity of.

    Returns
    -------
    list[dict[str,str]]
        The list of verdicts for each opinion. Each verdict is a dictionary with the "verdict" field.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": generate_toxicity_verdicts_instruction(
                opinions=opinions,
            ),
        },
    ]

    response = client(messages)
    response = trim_and_load_json(response)
    if type(response) != dict or "verdicts" not in response:
        raise InvalidLLMResponseError(
            f"LLM response was not a list of valid verdicts: {response}"
        )

    verdicts = response["verdicts"]
    if (
        type(verdicts) != list
        or len(verdicts) != len(opinions)
        or not all(verdict["verdict"] in ["yes", "no"] for verdict in verdicts)
    ):
        raise InvalidLLMResponseError(
            f"LLM response was not a list of valid verdicts: {response}"
        )

    return verdicts


def calculate_answer_correctness(
    client: ClientWrapper,
    system_prompt: str,
    query: str,
    response: str,
    groundtruths: list[str],
) -> float:
    prediction_statements = generate_statements(
        client=client,
        system_prompt=system_prompt,
        text=response,
    )
    f1_scores = []
    for groundtruth in groundtruths:
        groundtruth_statements = generate_statements(
            client=client,
            system_prompt=system_prompt,
            text=groundtruth,
        )
        verdicts = generate_answer_correctness_verdicts(
            client=client,
            system_prompt=system_prompt,
            query=query,
            groundtruth_statements=groundtruth_statements,
            prediction_statements=prediction_statements,
        )

        tp = len(verdicts["TP"])
        fp = len(verdicts["FP"])
        fn = len(verdicts["FN"])

        f1_scores.append(tp / (tp + 0.5 * (fp + fn)) if tp > 0 else 0)

    return max(f1_scores)


def calculate_answer_relevance(
    client: ClientWrapper,
    system_prompt: str,
    query: str,
    response: str,
) -> float:
    statements = generate_statements(
        client=client,
        system_prompt=system_prompt,
        text=response,
    )
    verdicts = generate_answer_relevance_verdicts(
        client=client,
        system_prompt=system_prompt,
        query=query,
        statements=statements,
    )
    return sum(1 for verdict in verdicts if verdict["verdict"] == "yes") / len(
        verdicts
    )


def calculate_bias(
    client: ClientWrapper,
    system_prompt: str,
    response: str,
) -> float:

    opinions = generate_opinions(
        client=client,
        system_prompt=system_prompt,
        text=response,
    )
    if len(opinions) == 0:
        return 0.0

    verdicts = generate_bias_verdicts(
        client=client, system_prompt=system_prompt, opinions=opinions
    )
    return sum(1 for verdict in verdicts if verdict["verdict"] == "yes") / len(
        verdicts
    )


def calculate_context_precision(
    client: ClientWrapper,
    system_prompt: str,
    query: str,
    retrieved_context: list[str],
    groundtruth_context: list[str],
) -> float:
    """
    Compute context precision, a score for evaluating the retrieval mechanism of a RAG model.

    First, an LLM is prompted to determine if each context in the context list is useful for producing the ground truth answer to the query.

    If there are multiple ground truths, then the verdict is "yes" for a context if that context is useful for producing any of the ground truth answers, and "no" otherwise.

    Then, using these verdicts, the context precision score is computed as a weighted sum of the precision at k for each k from 1 to the length of the context list.

    Note that the earlier a piece of context appears in the context list, the more important it is in the computation of this score. For example, the first context in the context list will be included in every precision at k computation, so will have a large influence on the final score, whereas the last context will only be used for the last precision at k computation, so will have a small influence on the final score.

    Parameters
    ----------
    query: Query
        A user query with ground truth and generated response.

    Returns
    -------
    Metric
        The context precision score between 0 and 1. A higher score indicates better context precision.
    """
    if len(retrieved_context) == 0:
        raise ValueError(
            "Context precision requires context in the prediction response."
        )
    if len(groundtruth_context) == 0:
        raise ValueError("Context precision requires ground truth context.")

    # Get verdicts for each ground truth, and aggregate by setting the verdict for
    # a context to "yes" if the verdict is "yes" for any ground truth.
    aggregate_verdicts = ["no"] * len(retrieved_context)
    for groundtruth in groundtruth_context:
        verdicts = generate_context_precision_verdicts(
            client=client,
            system_prompt=system_prompt,
            query=query,
            ordered_context_list=retrieved_context,
            groundtruth=groundtruth,
        )
        for i in range(len(verdicts)):
            if verdicts[i]["verdict"] == "yes":
                aggregate_verdicts[i] = "yes"

    # Use the aggregate verdicts to compute the precision at k for each k.
    precision_at_k_list = []
    for k in range(1, len(retrieved_context) + 1):
        # Only compute the precision at k if the kth context is relevant.
        if aggregate_verdicts[k - 1] == "yes":
            precision_at_k = (
                sum(
                    1 for verdict in aggregate_verdicts[:k] if verdict == "yes"
                )
                / k
            )
            precision_at_k_list.append(precision_at_k)

    # If none of the context are relevant, then the context precision is 0.
    if len(precision_at_k_list) == 0:
        return 0

    # Average over all the precision at k for which the kth context is relevant.
    return sum(precision_at_k_list) / len(precision_at_k_list)


def calculate_context_recall(
    client: ClientWrapper,
    system_prompt: str,
    retrieved_context: list[str],
    groundtruth_context: list[str],
) -> float:
    """
    Compute context recall, a score for evaluating the retrieval mechanism of a RAG model.

    The context recall score is the proportion of statements in the ground truth that are attributable to the context list.

    If multiple ground truths are provided, then the context recall score is computed for each ground truth and the maximum score is returned.

    Parameters
    ----------
    query: Query
        A user query with ground truth and generated response.

    Returns
    -------
    Metric
        The context recall score between 0 and 1. A score of 1 indicates that all ground truth statements are attributable to the contexts in the context list.
    """
    scores = []
    for groundtruth in groundtruth_context:
        groundtruth_statements = generate_statements(
            client=client, system_prompt=system_prompt, text=groundtruth
        )
        verdicts = generate_context_recall_verdicts(
            client=client,
            system_prompt=system_prompt,
            context_list=retrieved_context,
            groundtruth_statements=groundtruth_statements,
        )
        scores.append(
            sum(1 for verdict in verdicts if verdict["verdict"] == "yes")
            / len(verdicts)
        )

    return max(scores)


def calculate_context_relevance(
    client: ClientWrapper,
    system_prompt: str,
    query: str,
    context_prediction: list[str],
) -> float:
    """
    Compute context relevance, the proportion of contexts in the context list that are relevant to the query.

    Parameters
    ----------
    query: Query
        A user query with ground truth and generated response.

    Returns
    -------
    Metric
        The context relevance score between 0 and 1. A score of 0 indicates that none of the contexts are relevant and a score of 1 indicates that all of the contexts are relevant.
    """
    verdicts = generate_context_relevance_verdicts(
        client=client,
        system_prompt=system_prompt,
        query=query,
        context_list=context_prediction,
    )
    return sum(1 for verdict in verdicts if verdict["verdict"] == "yes") / len(
        verdicts
    )


def calculate_faithfulness(
    client: ClientWrapper,
    system_prompt: str,
    response: str,
    context_prediction: list[str],
) -> float:
    """
    Compute the faithfulness score. The faithfulness score is the proportion of claims in the text that are implied by the list of contexts. Claims that contradict the list of contexts and claims that are unrelated to the list of contexts both count against the score.

    Parameters
    ----------
    query: Query
        A user query with ground truth and generated response.

    Returns
    -------
    Metric
        The faithfulness score between 0 and 1. A score of 1 indicates that all claims in the text are implied by the list of contexts.
    """
    if len(context_prediction) == 0:
        raise ValueError(
            "Faithfulness requires context in the prediction response."
        )

    claims = generate_claims(
        client=client, system_prompt=system_prompt, text=response
    )

    # If there aren't any claims, then the text is perfectly faithful, as the text does not contain any non-faithful claims.
    if len(claims) == 0:
        return 1

    faithfulness_verdicts = generate_faithfulness_verdicts(
        client=client,
        system_prompt=system_prompt,
        claims=claims,
        context_list=context_prediction,
    )
    return sum(
        1 for verdict in faithfulness_verdicts if verdict["verdict"] == "yes"
    ) / len(faithfulness_verdicts)


def calculate_hallucination(
    client: ClientWrapper,
    system_prompt: str,
    response: str,
    context_prediction: list[str],
) -> float:
    """
    Compute the hallucination score, the proportion of contexts in the context list that are contradicted by the text.

    Parameters
    ----------
    query: Query
        A user query with ground truth and generated response.

    Returns
    -------
    Metric
        The hallucination score between 0 and 1. A score of 1 indicates that all contexts are contradicted by the text.
    """
    if len(context_prediction) == 0:
        raise ValueError(
            "Hallucination requires context in the prediction response."
        )

    verdicts = generate_hallucination_verdicts(
        client=client,
        system_prompt=system_prompt,
        text=response,
        context_list=context_prediction,
    )
    return sum(1 for verdict in verdicts if verdict["verdict"] == "yes") / len(
        verdicts
    )


def calculate_summary_coherence(
    client: ClientWrapper,
    system_prompt: str,
    text: str,
    summary: str,
) -> int:
    """
    Compute summary coherence, the collective quality of a summary.

    Parameters
    ----------
    text: str
        The text that was summarized.
    summary: str
        The summary to be evaluated.

    Returns
    -------
    int
        The summary coherence score will be evaluated as an integer, with 1 indicating the lowest summary coherence and 5 the highest summary coherence.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": generate_summary_coherence_instruction(
                text=text, summary=summary
            ),
        },
    ]

    response = client(messages)

    try:
        # Valid responses: "5", "\n5", "5\n", "5.", " 5", "5 {explanation}", etc.
        ret = int(response.strip()[0])
    except Exception:
        raise InvalidLLMResponseError(
            f"LLM response was not a valid summary coherence score: {response}"
        )

    if ret not in {1, 2, 3, 4, 5}:
        raise InvalidLLMResponseError(
            f"Summary coherence score was not an integer between 1 and 5: {ret}"
        )

    return ret


def calculate_toxicity(
    client: ClientWrapper,
    system_prompt: str,
    response: str,
) -> float:
    """
    Compute toxicity, the portion of opinions that are toxic.

    Parameters
    ----------
    query: Query
        A user query with ground truth and generated response.

    Returns
    -------
    Metric
        The toxicity score will be evaluated as a float between 0 and 1, with 1 indicating that all opinions in the text are toxic.
    """
    opinions = generate_opinions(
        client=client, system_prompt=system_prompt, text=response
    )
    if len(opinions) == 0:
        return 0.0

    verdicts = generate_toxicity_verdicts(
        client=client,
        system_prompt=system_prompt,
        opinions=opinions,
    )
    return sum(1 for verdict in verdicts if verdict["verdict"] == "yes") / len(
        verdicts
    )


def calculate_rouge_scores(
    prediction: str,
    references: list[str],
    rouge_types: list[str],
    use_stemmer: bool = False,
) -> dict[str, float]:
    """
    Calculate ROUGE scores for a prediction given some set of references.

    Computes scores using 'rouge1', 'rouge2', 'rougeL', and 'rougeLsum' where `rouge1`
    is unigram-based scoring, `rouge2` is bigram-based scoring, `rougeL` is scoring
    based on sentences (i.e., splitting on "." and ignoring "\n"), and `rougeLsum`
    is scoring based on splitting the text using "\n".

    Parameters
    ----------
    prediction : str
        A prediction to score. Each prediction should be a string with tokens separated by spaces.
    references : list[str]
        A list of reference for a given prediction. Each reference should be a string with tokens separated by spaces.
    rouge_types : list[str]
        A list of rouge types to calculate.
    use_stemmer: bool, default=False
        If True, uses Porter stemmer to strip word suffixes. Defaults to False.
    """
    rouge = evaluate.load("rouge")

    metrics = rouge.compute(
        predictions=[prediction],
        references=[references],
        rouge_types=rouge_types,
        use_stemmer=use_stemmer,
        use_aggregator=False,  # aggregation gives us an average across all predictions, which isn't what we want
    )

    if not metrics:
        return dict()

    # find the max value for each prediction
    results = defaultdict(float)
    for type_ in rouge_types:
        results[type_] = max(metrics[type_], results[type_])
    return results


def calculate_sentence_bleu(
    prediction: str,
    references: list[str],
    weights: list[float],
) -> float:
    """
    Calculate sentence BLEU scores for a of prediction - ground truth pair.

    Parameters
    ----------
    prediction : str
        The prediction to score. Each prediction should be a string with tokens separated by spaces.
    references : list[str]
        A list of references for the prediction. Each reference should be a string with tokens separated by spaces.
    weights : list[float]
        The default BLEU calculates a score for up to 4-grams using uniform
        weights (this is called BLEU-4). To evaluate your translations with
        higher/lower order ngrams, use customized weights. Example: when accounting
        for up to 5-grams with uniform weights (this is called BLEU-5) use [1/5]*5
    """
    if len(weights) == 0:
        raise ValueError("At least one weight should be defined.")

    tokenizer = RegexpTokenizer(
        r"\w+|\$[\d]+|[^\s\.]+"
    )  # regex tokenizer that ignores periods

    tokenized_prediction = tokenizer.tokenize(prediction)
    tokenized_references = [tokenizer.tokenize(ref) for ref in references]

    # find the max value for each prediction
    return max(
        float(
            bleu_score.sentence_bleu(
                references=tokenized_references,
                hypothesis=tokenized_prediction,
                weights=weights,
            ),  # type: ignore
        ),
        0.0,
    )
