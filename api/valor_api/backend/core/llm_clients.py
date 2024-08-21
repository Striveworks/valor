from typing import Any

from mistralai.sdk import Mistral
from openai import OpenAI
from pydantic import BaseModel

from valor_api.backend.core.llm_instructions_analysis import (
    generate_answer_correctness_verdicts_instruction,
    generate_answer_relevance_verdicts_instruction,
    generate_bias_verdicts_instruction,
    generate_claims_instruction,
    generate_coherence_instruction,
    generate_context_precision_verdicts_instruction,
    generate_context_recall_verdicts_instruction,
    generate_context_relevance_verdicts_instruction,
    generate_faithfulness_verdicts_instruction,
    generate_hallucination_verdicts_instruction,
    generate_opinions_instruction,
    generate_statements_instruction,
    generate_toxicity_verdicts_instruction,
)
from valor_api.backend.metrics.metric_utils import trim_and_load_json
from valor_api.exceptions import InvalidLLMResponseError

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class Message(BaseModel):
    role: str
    content: str


class Messages(BaseModel):
    messages: list[Message]


class LLMClient:
    """
    Parent class for all LLM clients.

    Attributes
    ----------
    api_key : str, optional
        The API key to use.
    model_name : str
        The model to use.
    """

    api_key: str | None = None
    model_name: str

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ):
        """
        Set the API key and model name (if provided).
        """
        self.api_key = api_key
        if model_name is not None:
            self.model_name = model_name

    def connect(
        self,
    ):
        """
        Setup the connection to the API. Not implemented for parent class.
        """
        raise NotImplementedError

    def _process_messages(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """
        Format messages for the API.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        Any
            The messages formatted for the API.
        """
        # Validate that the input is a list of dictionaries with "role" and "content" keys.
        _ = Messages(messages=messages)  # type: ignore

        raise NotImplementedError

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Call to the API. Not implemented for parent class.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        str
            The response from the API.
        """
        raise NotImplementedError

    def _generate_claims(
        self,
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
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": generate_claims_instruction(text=text),
            },
        ]

        response = self(messages)
        response = trim_and_load_json(response)
        if type(response) != dict or "claims" not in response:
            raise InvalidLLMResponseError(
                f"LLM response was not a dictionary or 'claims' was not in response: {response}"
            )
        claims = response["claims"]
        if type(claims) != list or not all(
            type(claim) == str for claim in claims
        ):
            raise InvalidLLMResponseError(
                f"LLM response was not a valid list of claims (list[str]): {response}"
            )
        return claims

    def _generate_opinions(
        self,
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
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": generate_opinions_instruction(text=text),
            },
        ]

        response = self(messages)
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

    def _generate_statements(
        self,
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
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": generate_statements_instruction(text=text),
            },
        ]

        response = self(messages)
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

    def _generate_answer_correctness_verdicts(
        self,
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
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": generate_answer_correctness_verdicts_instruction(
                    query=query,
                    prediction_statements=prediction_statements,
                    groundtruth_statements=groundtruth_statements,
                ),
            },
        ]
        response = self(messages)
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

        if len(response["TP"]) + len(response["FP"]) != len(
            prediction_statements
        ):
            raise InvalidLLMResponseError(
                f"Number of true positives and false positives did not match the number of prediction statements: {response}"
            )

        if len(response["FN"]) > len(groundtruth_statements):
            raise InvalidLLMResponseError(
                f"Number of false negatives exceeded the number of ground truth statements: {response}"
            )

        return response

    def _generate_answer_relevance_verdicts(
        self,
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
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": generate_answer_relevance_verdicts_instruction(
                    query=query,
                    statements=statements,
                ),
            },
        ]

        response = self(messages)
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
                verdict["verdict"] in ["yes", "no", "idk"]
                for verdict in verdicts
            )
        ):
            raise InvalidLLMResponseError(
                f"LLM response was not a list of valid verdicts: {response}"
            )

        return verdicts

    def _generate_bias_verdicts(
        self,
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
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": generate_bias_verdicts_instruction(
                    opinions=opinions,
                ),
            },
        ]

        response = self(messages)
        response = trim_and_load_json(response)
        if type(response) != dict or "verdicts" not in response:
            raise InvalidLLMResponseError(
                f"LLM response was not a list of valid verdicts: {response}"
            )

        verdicts = response["verdicts"]
        if (
            type(verdicts) != list
            or len(verdicts) != len(opinions)
            or not all(
                verdict["verdict"] in ["yes", "no"] for verdict in verdicts
            )
        ):
            raise InvalidLLMResponseError(
                f"LLM response was not a list of valid verdicts: {response}"
            )

        return verdicts

    def _coherence(
        self,
        text: str,
    ) -> int:
        """
        Compute coherence, the collective quality of all sentences, for a single piece of text.

        Parameters
        ----------
        text: str
            The text to be evaluated.

        Returns
        -------
        int
            The coherence score will be evaluated as an integer, with 1 indicating the lowest coherence and 5 the highest coherence.
        """
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": generate_coherence_instruction(text=text),
            },
        ]

        response = self(messages)

        try:
            # Valid responses: "5", "\n5", "5\n", "5.", " 5", "5 {explanation}", etc.
            ret = int(response.strip()[0])
        except Exception:
            raise InvalidLLMResponseError(
                f"LLM response was not a valid coherence score: {response}"
            )

        if ret not in {1, 2, 3, 4, 5}:
            raise InvalidLLMResponseError(
                f"Coherence score was not an integer between 1 and 5: {ret}"
            )

        return ret

    def _generate_context_precision_verdicts(
        self,
        query: str,
        context_list: list[str],
        groundtruth: str,
    ) -> list[dict[str, str]]:
        """
        Generate a list of context precision verdicts for a list of contexts, using a call to the LLM API.

        The verdict for each context should be 'yes' if the context is relevant to produce the ground truth answer to the query. The verdict should be 'no' otherwise.

        Parameters
        ----------
        query: str
            The query.
        context_list: list[str]
            The ordered list of contexts. Each context will be evaluated to determine if it is useful for producing the ground truth answer to the query.
        groundtruth: str
            The ground truth answer to the query.

        Returns
        -------
        list[dict[str,str]]
            The list of verdicts for each context. Each verdict is a dictionary with the "verdict" field.
        """
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": generate_context_precision_verdicts_instruction(
                    query=query,
                    context_list=context_list,
                    groundtruth=groundtruth,
                ),
            },
        ]

        response = self(messages)
        response = trim_and_load_json(response)
        if type(response) != dict or "verdicts" not in response:
            raise InvalidLLMResponseError(
                f"LLM response was not a list of valid verdicts: {response}"
            )

        verdicts = response["verdicts"]
        if (
            type(verdicts) != list
            or len(verdicts) != len(context_list)
            or not all(
                verdict["verdict"] in ["yes", "no"] for verdict in verdicts
            )
        ):
            raise InvalidLLMResponseError(
                f"LLM response was not a list of valid verdicts: {response}"
            )

        return verdicts

    def _generate_context_recall_verdicts(
        self,
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
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": generate_context_recall_verdicts_instruction(
                    context_list=context_list,
                    groundtruth_statements=groundtruth_statements,
                ),
            },
        ]

        response = self(messages)
        response = trim_and_load_json(response)
        if type(response) != dict or "verdicts" not in response:
            raise InvalidLLMResponseError(
                f"LLM response was not a list of valid verdicts: {response}"
            )

        verdicts = response["verdicts"]
        if (
            type(verdicts) != list
            or len(verdicts) != len(groundtruth_statements)
            or not all(
                verdict["verdict"] in ["yes", "no"] for verdict in verdicts
            )
        ):
            raise InvalidLLMResponseError(
                f"LLM response was not a list of valid verdicts: {response}"
            )

        return verdicts

    def _generate_context_relevance_verdicts(
        self,
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
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": generate_context_relevance_verdicts_instruction(
                    query=query,
                    context_list=context_list,
                ),
            },
        ]

        response = self(messages)
        response = trim_and_load_json(response)
        if type(response) != dict or "verdicts" not in response:
            raise InvalidLLMResponseError(
                f"LLM response was not a list of valid verdicts: {response}"
            )

        verdicts = response["verdicts"]
        if (
            type(verdicts) != list
            or len(verdicts) != len(context_list)
            or not all(
                verdict["verdict"] in ["yes", "no"] for verdict in verdicts
            )
        ):
            raise InvalidLLMResponseError(
                f"LLM response was not a list of valid verdicts: {response}"
            )

        return verdicts

    def _generate_faithfulness_verdicts(
        self,
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
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": generate_faithfulness_verdicts_instruction(
                    claims=claims,
                    context_list=context_list,
                ),
            },
        ]

        response = self(messages)
        response = trim_and_load_json(response)
        if type(response) != dict or "verdicts" not in response:
            raise InvalidLLMResponseError(
                f"LLM response was not a list of valid verdicts: {response}"
            )

        verdicts = response["verdicts"]
        if (
            type(verdicts) != list
            or len(verdicts) != len(claims)
            or not all(
                verdict["verdict"] in ["yes", "no"] for verdict in verdicts
            )
        ):
            raise InvalidLLMResponseError(
                f"LLM response was not a list of valid verdicts: {response}"
            )

        return verdicts

    def _generate_hallucination_verdicts(
        self,
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
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": generate_hallucination_verdicts_instruction(
                    text=text,
                    context_list=context_list,
                ),
            },
        ]

        response = self(messages)
        response = trim_and_load_json(response)
        if type(response) != dict or "verdicts" not in response:
            raise InvalidLLMResponseError(
                f"LLM response was not a list of valid verdicts: {response}"
            )

        verdicts = response["verdicts"]
        if (
            type(verdicts) != list
            or len(verdicts) != len(context_list)
            or not all(
                verdict["verdict"] in ["yes", "no"] for verdict in verdicts
            )
        ):
            raise InvalidLLMResponseError(
                f"LLM response was not a list of valid verdicts: {response}"
            )

        return verdicts

    def _generate_toxicity_verdicts(
        self,
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
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": generate_toxicity_verdicts_instruction(
                    opinions=opinions,
                ),
            },
        ]

        response = self(messages)
        response = trim_and_load_json(response)
        if type(response) != dict or "verdicts" not in response:
            raise InvalidLLMResponseError(
                f"LLM response was not a list of valid verdicts: {response}"
            )

        verdicts = response["verdicts"]
        if (
            type(verdicts) != list
            or len(verdicts) != len(opinions)
            or not all(
                verdict["verdict"] in ["yes", "no"] for verdict in verdicts
            )
        ):
            raise InvalidLLMResponseError(
                f"LLM response was not a list of valid verdicts: {response}"
            )

        return verdicts

    def answer_correctness(
        self,
        query: str,
        prediction: str,
        groundtruth: str,
    ) -> float:
        """
        Compute answer correctness. Answer correctness is computed as an f1 score obtained by comparing prediction statements to ground truth statements.

        This metric was adapted from RAGAS. We follow a similar prompting strategy and computation, however we do not do a weighted sum with an answer similarity score using embeddings.

        Parameters
        ----------
        query: str
            The query that both the ground truth and prediction should be answering.
        prediction: str
            The prediction text to extract statements from.
        groundtruth: str
            The ground truth text to extract statements from.

        Returns
        -------
        float
            The answer correctness score between 0 and 1, with higher values indicating that the answer is more correct. A score of 1 indicates that all statements in the prediction are supported by the ground truth and all statements in the ground truth are present in the prediction.
        """
        groundtruth_statements = self._generate_statements(groundtruth)
        prediction_statements = self._generate_statements(prediction)
        verdicts = self._generate_answer_correctness_verdicts(
            query, groundtruth_statements, prediction_statements
        )

        tp = len(verdicts["TP"])
        fp = len(verdicts["FP"])
        fn = len(verdicts["FN"])

        f1_score = tp / (tp + 0.5 * (fp + fn)) if tp > 0 else 0
        return f1_score

    def answer_relevance(
        self,
        query: str,
        text: str,
    ) -> float:
        """
        Compute answer relevance, the proportion of statements that are relevant to the query, for a single piece of text.

        Parameters
        ----------
        query: str
            The query to evaluate the statements against.
        text: str
            The text to extract statements from.

        Returns
        -------
        float
            The answer relevance score will be evaluated as a float between 0 and 1, with 1 indicating that all statements are relevant to the query.
        """
        statements = self._generate_statements(text)
        verdicts = self._generate_answer_relevance_verdicts(query, statements)
        return sum(
            1 for verdict in verdicts if verdict["verdict"] == "yes"
        ) / len(verdicts)

    def bias(
        self,
        text: str,
    ) -> float:
        """
        Compute bias, the portion of opinions that are biased.

        Parameters
        ----------
        text: str
            The text to be evaluated.

        Returns
        -------
        float
            The bias score will be evaluated as a float between 0 and 1, with 1 indicating that all opinions in the text are biased.
        """
        opinions = self._generate_opinions(text)
        if len(opinions) == 0:
            return 0

        verdicts = self._generate_bias_verdicts(opinions)

        return sum(
            1 for verdict in verdicts if verdict["verdict"] == "yes"
        ) / len(verdicts)

    def coherence(
        self,
        text: str,
    ) -> int:
        """
        Compute coherence, the collective quality of all sentences, for a single piece of text.

        Parameters
        ----------
        text: str
            The text to be evaluated.

        Returns
        -------
        int
            The coherence score will be evaluated as an integer, with 1 indicating the lowest coherence and 5 the highest coherence.
        """
        return self._coherence(text)

    def context_precision(
        self,
        query: str,
        context_list: list[str],
        groundtruth: str,
    ) -> float:
        """
        Compute context precision, a score for evaluating the retrieval mechanism of a RAG model.

        First, an LLM is prompted to determine if each context in the context list is useful for producing the ground truth answer to the query.

        Then the context precision score is computed as a weighted sum of the precision at k for each k from 1 to the length of the context list.

        Note that the earlier a piece of context appears in the context list, the more important it is in the computation of this score. For example, the first context in the context list will be included in every precision at k computation, so will have a large influence on the final score, whereas the last context will only be used for the last precision at k computation, so will have a small influence on the final score.

        Parameters
        ----------
        query: str
            A query.
        context_list: list[str]
            The ordered list of contexts. Each context will be evaluated to determine if it is useful for producing the ground truth answer to the query. Contexts in this list are NOT treated equally in the computation of this score. The earlier a piece of context appears in the context list, the more important it is in the computation of this score.
        groundtruth: str
            A ground truth answer to the query.

        Returns
        -------
        float
            The context precision score will be evaluated as a float between 0 and 1, with a higher score indicating better context precision.
        """
        if len(context_list) == 0:
            raise ValueError(
                "Context precision is meaningless if context_list is empty."
            )

        verdicts = self._generate_context_precision_verdicts(
            query, context_list, groundtruth
        )

        precision_at_k_list = []
        for k in range(1, len(context_list) + 1):
            # Only compute the precision at k if the kth context is relevant.
            if verdicts[k - 1]["verdict"] == "yes":
                precision_at_k = (
                    sum(
                        1
                        for verdict in verdicts[:k]
                        if verdict["verdict"] == "yes"
                    )
                    / k
                )
                precision_at_k_list.append(precision_at_k)

        # If none of the context are relevant, then the context precision is 0.
        if len(precision_at_k_list) == 0:
            return 0

        # Average over all the precision at k for which the kth context is relevant.
        return sum(precision_at_k_list) / len(precision_at_k_list)

    def context_recall(
        self,
        context_list: list[str],
        groundtruth: str,
    ) -> float:
        """
        Compute context recall, a score for evaluating the retrieval mechanism of a RAG model.

        The context recall score is the proportion of statements in the ground truth that are attributable to the context list.

        Parameters
        ----------
        context_list: list[str]
            The list of contexts to evaluate against.
        groundtruth: str
            A ground truth answer to extract statements from.

        Returns
        -------
        float
            The context recall score will be evaluated as a float between 0 and 1, with 1 indicating that all ground truth statements are attributable to the context list.
        """
        if len(context_list) == 0:
            raise ValueError(
                "Context recall is meaningless if context_list is empty."
            )

        groundtruth_statements = self._generate_statements(groundtruth)

        verdicts = self._generate_context_recall_verdicts(
            context_list, groundtruth_statements
        )

        return sum(
            1 for verdict in verdicts if verdict["verdict"] == "yes"
        ) / len(verdicts)

    def context_relevance(
        self,
        query: str,
        context_list: list[str],
    ) -> float:
        """
        Compute context relevance, the proportion of contexts in the context list that are relevant to the query.

        Parameters
        ----------
        query: str
            The query to evaluate each context against.
        context_list: list[str]
            The list of contexts to evaluate the relevance of.

        Returns
        -------
        float
            The context relevance score will be evaluated as a float between 0 and 1, with 0 indicating that none of the contexts are relevant and 1 indicating that all of the contexts are relevant.
        """
        if len(context_list) == 0:
            raise ValueError(
                "Context relevance is meaningless if context_list is empty."
            )

        verdicts = self._generate_context_relevance_verdicts(
            query, context_list
        )

        return sum(
            1 for verdict in verdicts if verdict["verdict"] == "yes"
        ) / len(verdicts)

    def faithfulness(
        self,
        text: str,
        context_list: list[str],
    ) -> float:
        """
        Compute the faithfulness score. The faithfulness score is the proportion of claims in the text that are implied by the list of contexts. Claims that contradict the list of contexts and claims that are unrelated to the list of contexts both count against the score.

        Parameters
        ----------
        text: str
            The text to evaluate for faithfulness.
        context_list: list[str]
            The list of contexts to compare against.

        Returns
        -------
        float
            The faithfulness score will be evaluated as a float between 0 and 1, with 1 indicating that all claims in the text are implied by the list of contexts.
        """
        if len(context_list) == 0:
            raise ValueError(
                "Faithfulness is meaningless if context_list is empty."
            )

        claims = self._generate_claims(text)

        # If there aren't any claims, then the text is perfectly faithful, as the text does not contain any non-faithful claims.
        if len(claims) == 0:
            return 1

        faithfulness_verdicts = self._generate_faithfulness_verdicts(
            claims, context_list
        )

        return sum(
            1
            for verdict in faithfulness_verdicts
            if verdict["verdict"] == "yes"
        ) / len(faithfulness_verdicts)

    def hallucination(
        self,
        text: str,
        context_list: list[str],
    ) -> float:
        """
        Compute the hallucination score, the proportion of contexts in the context list that are contradicted by the text.

        Parameters
        ----------
        text: str
            The text to evaluate for hallucination.
        context_list: list[str]
            The list of contexts to compare against.

        Returns
        -------
        float
            The hallucination score will be evaluated as a float between 0 and 1, with 1 indicating that all contexts are contradicted by the text.
        """
        if len(context_list) == 0:
            raise ValueError(
                "Hallucination is meaningless if context_list is empty."
            )

        verdicts = self._generate_hallucination_verdicts(
            text,
            context_list,
        )

        return sum(
            1 for verdict in verdicts if verdict["verdict"] == "yes"
        ) / len(verdicts)

    def toxicity(
        self,
        text: str,
    ) -> float:
        """
        Compute toxicity, the portion of opinions that are toxic.

        Parameters
        ----------
        text: str
            The text to be evaluated.

        Returns
        -------
        float
            The toxicity score will be evaluated as a float between 0 and 1, with 1 indicating that all opinions in the text are toxic.
        """
        opinions = self._generate_opinions(text)
        if len(opinions) == 0:
            return 0

        verdicts = self._generate_toxicity_verdicts(opinions)

        return sum(
            1 for verdict in verdicts if verdict["verdict"] == "yes"
        ) / len(verdicts)


class WrappedOpenAIClient(LLMClient):
    """
    Wrapper for calls to OpenAI's API.

    Attributes
    ----------
    api_key : str, optional
        The OpenAI API key to use. If not specified, then the OPENAI_API_KEY environment variable will be used.
    seed : int, optional
        An optional seed can be provided to GPT to get deterministic results.
    model_name : str
        The model to use. Defaults to "gpt-3.5-turbo".
    """

    api_key: str | None = None
    seed: int | None = None
    model_name: str = "gpt-3.5-turbo"
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def __init__(
        self,
        api_key: str | None = None,
        seed: int | None = None,
        model_name: str | None = None,
    ):
        """
        Set the API key, seed and model name (if provided).
        """
        self.api_key = api_key
        if seed is not None:
            self.seed = seed
        if model_name is not None:
            self.model_name = model_name

    def connect(
        self,
    ):
        """
        Setup the connection to the API.
        """
        if self.api_key is None:
            self.client = OpenAI()
        else:
            self.client = OpenAI(api_key=self.api_key)

    def _process_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        Format messages for the API.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        list[dict[str, str]]
            The messages are left in the OpenAI standard.
        """
        # Validate that the input is a list of dictionaries with "role" and "content" keys.
        _ = Messages(messages=messages)  # type: ignore

        return messages

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Call to the API.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        str
            The response from the API.
        """
        processed_messages = self._process_messages(messages)
        openai_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=processed_messages,  # type: ignore - mistralai issue
            seed=self.seed,
        )

        response = openai_response.choices[0].message.content
        if openai_response.usage is not None:
            self.total_prompt_tokens += openai_response.usage.prompt_tokens
            self.total_completion_tokens += (
                openai_response.usage.completion_tokens
            )
        finish_reason = openai_response.choices[
            0
        ].finish_reason  # Enum: "stop" "length" "content_filter" "tool_calls" "function_call"

        if finish_reason == "length":
            raise ValueError(
                "OpenAI response reached max token limit. Resulting evaluation is likely invalid or of low quality."
            )
        elif finish_reason == "content_filter":
            raise ValueError(
                "OpenAI response was flagged by content filter. Resulting evaluation is likely invalid or of low quality."
            )

        if response is None:
            response = ""
        return response


class WrappedMistralAIClient(LLMClient):
    """
    Wrapper for calls to Mistral's API.

    Attributes
    ----------
    api_key : str, optional
        The Mistral API key to use. If not specified, then the MISTRAL_API_KEY environment variable will be used.
    model_name : str
        The model to use. Defaults to "mistral-small-latest".
    """

    api_key: str | None = None
    model_name: str = "mistral-small-latest"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ):
        """
        Set the API key and model name (if provided).
        """
        self.api_key = api_key
        if model_name is not None:
            self.model_name = model_name

    def connect(
        self,
    ):
        """
        Setup the connection to the API.
        """
        if self.api_key is None:
            self.client = Mistral()
        else:
            self.client = Mistral(api_key=self.api_key)

    def _process_messages(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """
        Format messages for Mistral's API.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        Any
            The messages formatted for Mistral's API. With mistralai>=1.0.0, the messages can be left in the OpenAI standard.
        """
        # Validate that the input is a list of dictionaries with "role" and "content" keys.
        _ = Messages(messages=messages)  # type: ignore

        return messages

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Call to the API.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        str
            The response from the API.
        """
        processed_messages = self._process_messages(messages)
        mistral_response = self.client.chat.complete(
            model=self.model_name,
            messages=processed_messages,
        )
        if mistral_response is None or mistral_response.choices is None:
            return ""

        finish_reason = mistral_response.choices[
            0
        ].finish_reason  # Enum: "stop" "length" "model_length" "error" "tool_calls"
        if mistral_response.choices[0].message is None:
            response = ""
        else:
            response = mistral_response.choices[0].message.content

        if finish_reason == "length":
            raise ValueError(
                "Mistral response reached max token limit. Resulting evaluation is likely invalid or of low quality."
            )

        if not isinstance(response, str):
            raise TypeError("Mistral AI response was not a string.")

        return response


class MockLLMClient(LLMClient):
    """
    A mocked LLM client for testing purposes.

    Attributes
    ----------
    api_key : str, optional
        The API key to use.
    model_name : str
        The model to use. A model_name is not required for testing purposes.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Neither the api_key nor the model_name are required for the mock client.
        """
        pass

    def connect(
        self,
    ):
        """
        No connection is required for the mock client.
        """
        pass

    def _process_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        Format messages for the API.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        list[dict[str, str]]
            The messages are left in the OpenAI format.
        """
        # Validate that the input is a list of dictionaries with "role" and "content" keys.
        _ = Messages(messages=messages)  # type: ignore

        return messages

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Call to the API. Returns "" by default, or metric specific mock responses.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        str
            The response from the API.
        """
        response = None

        processed_messages = self._process_messages(messages)
        if len(processed_messages) >= 2:
            # Generate claims
            if (
                "generate a comprehensive list of FACTUAL CLAIMS"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "claims": [
            "The capital of the UK is London.",
            "The capital of South Korea is Seoul.",
            "The capital of Argentina is Canada."
        ]
    }```"""

            # Generate opinions
            elif (
                "generate a list of OPINIONS"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "opinions": [
            "I like the color green.",
            "People from Canada are nicer than people from other countries."
        ]
    }```"""

            # Generate statements
            elif (
                "generate a list of STATEMENTS"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "statements": [
            "The capital of the UK is London.",
            "London is the largest city in the UK by population and GDP."
        ]
    }```"""

            # Answer correctness verdicts
            elif (
                "Return in JSON format with three keys: 'TP', 'FP', and 'FN'"
                in processed_messages[1]["content"]
            ):
                response = """```json
{
    "TP": [
        "London is the largest city in the UK by GDP"
    ],
    "FP": [
        "London is the largest city in the UK by population"
    ],
    "FN": [
        "In 2021, financial services made up more than 20% of London's output"
    ]
}```"""

            # Answer relevance verdicts
            elif (
                "generate a list of verdicts that indicate whether each statement is relevant to address the query"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "verdicts": [
            {"verdict": "yes"},
            {"verdict": "no"}
        ]
    }```"""

            # Bias verdicts
            elif (
                "generate a list of verdicts to indicate whether EACH opinion is biased"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "verdicts": [
            {"verdict": "no"},
            {"verdict": "yes"}
        ]
    }```"""

            # Coherence score
            elif (
                "Coherence (1-5) - the collective quality of all sentences."
                in processed_messages[1]["content"]
            ):
                response = "4"

            # Context precision verdicts
            elif (
                "generate a list of verdicts to determine whether each context in the context list is useful for producing the ground truth answer to the query"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "verdicts": [
            {"verdict": "yes"},
            {"verdict": "no"},
            {"verdict": "no"},
            {"verdict": "yes"}
        ]
    }```"""

            # Context recall verdicts
            elif (
                "analyze each ground truth statement and determine if the statement can be attributed to the given context"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "verdicts": [
            {"verdict": "yes"},
            {"verdict": "yes"}
        ]
    }```"""

            # Context relevance verdicts
            elif (
                "generate a list of verdicts to indicate whether each context is relevant to the provided query"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "verdicts": [
            {"verdict": "yes"},
            {"verdict": "yes"},
            {"verdict": "no"},
            {"verdict": "yes"}
        ]
    }```"""

            # Faithfulness verdicts
            elif (
                "generate a list of verdicts to indicate whether EACH claim is implied by the context list"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "verdicts": [
            {"verdict": "yes"},
            {"verdict": "no"},
            {"verdict": "no"}
        ]
    }```"""

            # Hallucination agreement verdicts
            elif (
                "generate a list of verdicts to indicate whether the given text contradicts EACH context"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "verdicts": [
            {"verdict": "no"},
            {"verdict": "no"},
            {"verdict": "yes"},
            {"verdict": "no"}
        ]
    }```"""

            # Toxicity verdicts
            elif (
                "generate a list of verdicts to indicate whether EACH opinion is toxic"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "verdicts": [
            {"verdict": "no"},
            {"verdict": "no"}
        ]
    }```"""

        if response is None:
            response = ""
        return response
