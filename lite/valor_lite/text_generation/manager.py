from functools import wraps

from valor_lite.text_generation.annotation import Query
from valor_lite.text_generation.computation import (
    calculate_rouge_scores,
    calculate_sentence_bleu,
)
from valor_lite.text_generation.exceptions import InvalidLLMResponseError
from valor_lite.text_generation.llm_client import (
    MistralWrapper,
    MockWrapper,
    OpenAIWrapper,
    _ClientWrapper,
)
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
from valor_lite.text_generation.metric import Metric, MetricType, ROUGEType
from valor_lite.text_generation.utilities import trim_and_load_json


def llm_guided(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if self.client is None:
            raise ValueError(
                f"{fn.__name__} requires the definition of an LLM client."
            )
        return fn(self, *args, **kwargs)

    return wrapper


def retry_if_invalid_llm_response(fn):
    """
    Call the LLMClient class function with retries for InvalidLLMResponseError.

    If retries is set to 0, then the function will only be called once and not retried.

    If, for example, retries is set to 3, then the function will be retried in the event of an InvalidLLMResponseError up to 3 times, for a maximum of 4 calls.
    """

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if self.client is None:
            raise ValueError(
                f"{fn.__name__} requires the definition of an LLM client."
            )
        error = None
        retries = getattr(self, "retries", 0)
        for _ in range(1 + retries):
            try:
                return fn(self, *args, **kwargs)
            except InvalidLLMResponseError as e:
                error = e
        if error is not None:
            return Metric.error(
                error_type=type(error).__name__,
                error_message=f"Failed after {retries} retries.",
            )

    return wrapper


class Evaluator:
    """
    Parent class for all LLM clients.

    Attributes
    ----------
    client : _ClientWrapper, optional
        An optional client to compute llm-guided metrics.
    retries : int
        The number of times to retry the API call if it fails. Defaults to 0, indicating that the call will not be retried.
    """

    def __init__(
        self,
        client: _ClientWrapper | None = None,
        retries: int = 0,
    ):
        """
        Creates an instance of a generic LLM client.

        Parameters
        ----------
        client : _ClientWrapper, optional
            Any LLM client that conforms to _ClientWrapper. Required for LLM-guided metrics.
        retries : int, default=0
            The number of times to retry the API call if it fails. Defaults to 0, indicating that the call will not be retried.
        """
        self.client = client
        self.retries = retries
        self.default_system_prompt = "You are a helpful assistant."

    @classmethod
    def openai(
        cls,
        model_name: str = "gpt-3.5-turbo",
        api_key: str | None = None,
        retries: int = 0,
        seed: int | None = None,
    ):
        """
        Create an evaluator using OpenAI's client.

        Parameters
        ----------
        model_name : str, default="gpt-3.5-turbo"
            The model to use. Defaults to "gpt-3.5-turbo".
        api_key : str, optional
            The OpenAI API key to use. If not specified, then the OPENAI_API_KEY environment variable will be used.
        retries : int, default=0
            The number of times to retry the API call if it fails. Defaults to 0, indicating that the call will not be retried. For example, if self.retries is set to 3, this means that the call will be retried up to 3 times, for a maximum of 4 calls.
        seed : int, optional
            An optional seed can be provided to GPT to get deterministic results.
        """
        if seed is not None:
            if retries != 0:
                raise ValueError(
                    "Seed is provided, but retries is not 0. Retries should be 0 when seed is provided."
                )
        client = OpenAIWrapper(
            api_key=api_key,
            model_name=model_name,
            seed=seed,
        )
        return cls(client=client, retries=retries)

    @classmethod
    def mistral(
        cls,
        model_name: str = "mistral-small-latest",
        api_key: str | None = None,
        retries: int = 0,
    ):
        """
        Create an evaluator using the Mistral API.

        Parameters
        ----------
        model_name : str, default="mistral-small-latest"
            The model to use. Defaults to "mistral-small-latest".
        api_key : str, optional
            The Mistral API key to use. If not specified, then the MISTRAL_API_KEY environment variable will be used.
        retries : int, default=0
            The number of times to retry the API call if it fails. Defaults to 0, indicating that the call will not be retried. For example, if self.retries is set to 3, this means that the call will be retried up to 3 times, for a maximum of 4 calls.
        """
        client = MistralWrapper(
            api_key=api_key,
            model_name=model_name,
        )
        return cls(client=client, retries=retries)

    @classmethod
    def mock(
        cls,
        retries: int = 0,
        **kwargs,
    ):
        """
        Create an evaluator with a mocked LLM client.
        """
        client = MockWrapper(**kwargs, retries=retries)
        return cls(client=client)

    @retry_if_invalid_llm_response
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

        if self.client is None:
            raise ValueError(
                "A LLM client must be defined to use this function."
            )

        messages = [
            {"role": "system", "content": self.default_system_prompt},
            {
                "role": "user",
                "content": generate_claims_instruction(text=text),
            },
        ]

        response = self.client(messages)
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

    @retry_if_invalid_llm_response
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

        if self.client is None:
            raise ValueError(
                "A LLM client must be defined to use this function."
            )

        messages = [
            {"role": "system", "content": self.default_system_prompt},
            {
                "role": "user",
                "content": generate_opinions_instruction(text=text),
            },
        ]

        response = self.client(messages)
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

    @retry_if_invalid_llm_response
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

        if self.client is None:
            raise ValueError(
                "A LLM client must be defined to use this function."
            )

        messages = [
            {"role": "system", "content": self.default_system_prompt},
            {
                "role": "user",
                "content": generate_statements_instruction(text=text),
            },
        ]

        response = self.client(messages)
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

    @retry_if_invalid_llm_response
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

        if self.client is None:
            raise ValueError(
                "A LLM client must be defined to use this function."
            )

        messages = [
            {"role": "system", "content": self.default_system_prompt},
            {
                "role": "user",
                "content": generate_answer_correctness_verdicts_instruction(
                    query=query,
                    prediction_statements=prediction_statements,
                    groundtruth_statements=groundtruth_statements,
                ),
            },
        ]
        response = self.client(messages)
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

    @retry_if_invalid_llm_response
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
        if self.client is None:
            raise ValueError(
                "A LLM client must be defined to use this function."
            )

        messages = [
            {"role": "system", "content": self.default_system_prompt},
            {
                "role": "user",
                "content": generate_answer_relevance_verdicts_instruction(
                    query=query,
                    statements=statements,
                ),
            },
        ]

        response = self.client(messages)
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

    @retry_if_invalid_llm_response
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

        if self.client is None:
            raise ValueError(
                "A LLM client must be defined to use this function."
            )

        messages = [
            {"role": "system", "content": self.default_system_prompt},
            {
                "role": "user",
                "content": generate_bias_verdicts_instruction(
                    opinions=opinions,
                ),
            },
        ]

        response = self.client(messages)
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

    @retry_if_invalid_llm_response
    def _generate_context_precision_verdicts(
        self,
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

        if self.client is None:
            raise ValueError(
                "A LLM client must be defined to use this function."
            )

        messages = [
            {"role": "system", "content": self.default_system_prompt},
            {
                "role": "user",
                "content": generate_context_precision_verdicts_instruction(
                    query=query,
                    ordered_context_list=ordered_context_list,
                    groundtruth=groundtruth,
                ),
            },
        ]

        response = self.client(messages)
        response = trim_and_load_json(response)
        if type(response) != dict or "verdicts" not in response:
            raise InvalidLLMResponseError(
                f"LLM response was not a list of valid verdicts: {response}"
            )

        verdicts = response["verdicts"]
        if (
            type(verdicts) != list
            or len(verdicts) != len(ordered_context_list)
            or not all(
                verdict["verdict"] in ["yes", "no"] for verdict in verdicts
            )
        ):
            raise InvalidLLMResponseError(
                f"LLM response was not a list of valid verdicts: {response}"
            )

        return verdicts

    @retry_if_invalid_llm_response
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

        if self.client is None:
            raise ValueError(
                "A LLM client must be defined to use this function."
            )

        messages = [
            {"role": "system", "content": self.default_system_prompt},
            {
                "role": "user",
                "content": generate_context_recall_verdicts_instruction(
                    context_list=context_list,
                    groundtruth_statements=groundtruth_statements,
                ),
            },
        ]

        response = self.client(messages)
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

    @retry_if_invalid_llm_response
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

        if self.client is None:
            raise ValueError(
                "A LLM client must be defined to use this function."
            )

        messages = [
            {"role": "system", "content": self.default_system_prompt},
            {
                "role": "user",
                "content": generate_context_relevance_verdicts_instruction(
                    query=query,
                    context_list=context_list,
                ),
            },
        ]

        response = self.client(messages)
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

    @retry_if_invalid_llm_response
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

        if self.client is None:
            raise ValueError(
                "A LLM client must be defined to use this function."
            )

        messages = [
            {"role": "system", "content": self.default_system_prompt},
            {
                "role": "user",
                "content": generate_faithfulness_verdicts_instruction(
                    claims=claims,
                    context_list=context_list,
                ),
            },
        ]

        response = self.client(messages)
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

    @retry_if_invalid_llm_response
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

        if self.client is None:
            raise ValueError(
                "A LLM client must be defined to use this function."
            )

        messages = [
            {"role": "system", "content": self.default_system_prompt},
            {
                "role": "user",
                "content": generate_hallucination_verdicts_instruction(
                    text=text,
                    context_list=context_list,
                ),
            },
        ]

        response = self.client(messages)
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

    @retry_if_invalid_llm_response
    def _summary_coherence(
        self,
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

        if self.client is None:
            raise ValueError(
                "A LLM client must be defined to use this function."
            )

        messages = [
            {"role": "system", "content": self.default_system_prompt},
            {
                "role": "user",
                "content": generate_summary_coherence_instruction(
                    text=text, summary=summary
                ),
            },
        ]

        response = self.client(messages)

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

    @retry_if_invalid_llm_response
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

        if self.client is None:
            raise ValueError(
                "A LLM client must be defined to use this function."
            )

        messages = [
            {"role": "system", "content": self.default_system_prompt},
            {
                "role": "user",
                "content": generate_toxicity_verdicts_instruction(
                    opinions=opinions,
                ),
            },
        ]

        response = self.client(messages)
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

    @llm_guided
    def compute_answer_correctness(
        self,
        query: Query,
    ) -> Metric:
        """
        Compute answer correctness. Answer correctness is computed as an f1 score obtained by comparing prediction statements to ground truth statements.

        If there are multiple ground truths, then the f1 score is computed for each ground truth and the maximum score is returned.

        This metric was adapted from RAGAS. We follow a similar prompting strategy and computation, however we do not do a weighted sum with an answer similarity score using embeddings.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.

        Returns
        -------
        Metric
            The answer correctness score between 0 and 1. Higher values indicate that the answer is more correct. A score of 1 indicates that all statements in the prediction are supported by the ground truth and all statements in the ground truth are present in the prediction.
        """
        if len(query.groundtruths) == 0:
            raise ValueError(
                "Answer correctness requires the definition of ground truth context."
            )

        prediction_statements = self._generate_statements(
            text=query.prediction.output
        )
        f1_scores = []
        for groundtruth in query.groundtruths:
            groundtruth_statements = self._generate_statements(
                text=groundtruth
            )
            verdicts = self._generate_answer_correctness_verdicts(
                query=query.query,
                groundtruth_statements=groundtruth_statements,
                prediction_statements=prediction_statements,
            )

            tp = len(verdicts["TP"])
            fp = len(verdicts["FP"])
            fn = len(verdicts["FN"])

            f1_scores.append(tp / (tp + 0.5 * (fp + fn)) if tp > 0 else 0)

        max_f1 = max(f1_scores)

        return Metric.answer_correctness(value=max_f1)

    @llm_guided
    def compute_answer_relevance(self, query: Query) -> Metric:
        """
        Compute answer relevance, the proportion of statements that are relevant to the query, for a single piece of text.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.

        Returns
        -------
        Metric
            The answer relevance score between 0 and 1. A score of 1 indicates that all statements are relevant to the query.
        """
        statements = self._generate_statements(text=query.prediction.output)
        verdicts = self._generate_answer_relevance_verdicts(
            query=query.query,
            statements=statements,
        )
        result = sum(
            1 for verdict in verdicts if verdict["verdict"] == "yes"
        ) / len(verdicts)
        return Metric.answer_relevance(value=result)

    @llm_guided
    def compute_bias(
        self,
        query: Query,
    ) -> Metric:
        """
        Compute bias, the portion of opinions that are biased.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.

        Returns
        -------
        float
            The bias score between 0 and 1. A score of 1 indicates that all opinions in the text are biased.
        """
        opinions = self._generate_opinions(text=query.prediction.output)
        if len(opinions) == 0:
            return 0.0

        verdicts = self._generate_bias_verdicts(opinions=opinions)

        result = sum(
            1 for verdict in verdicts if verdict["verdict"] == "yes"
        ) / len(verdicts)

        return Metric.bias(value=result)

    @llm_guided
    def compute_context_precision(
        self,
        query: Query,
    ) -> Metric:
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
        if len(query.prediction.context) == 0:
            raise ValueError(
                "Context precision requires context in the prediction response."
            )
        if len(query.groundtruth) == 0:
            raise ValueError(
                "Context precision requires ground truth contexts."
            )

        # Get verdicts for each ground truth, and aggregate by setting the verdict for
        # a context to "yes" if the verdict is "yes" for any ground truth.
        aggregate_verdicts = ["no"] * len(query.prediction.context)
        for groundtruth in query.groundtruth:
            verdicts = self._generate_context_precision_verdicts(
                query=query,
                ordered_context_list=query.prediction.context,
                groundtruth=groundtruth,
            )
            for i in range(len(verdicts)):
                if verdicts[i]["verdict"] == "yes":
                    aggregate_verdicts[i] = "yes"

        # Use the aggregate verdicts to compute the precision at k for each k.
        precision_at_k_list = []
        for k in range(1, len(query.prediction.context) + 1):
            # Only compute the precision at k if the kth context is relevant.
            if aggregate_verdicts[k - 1] == "yes":
                precision_at_k = (
                    sum(
                        1
                        for verdict in aggregate_verdicts[:k]
                        if verdict == "yes"
                    )
                    / k
                )
                precision_at_k_list.append(precision_at_k)

        # If none of the context are relevant, then the context precision is 0.
        if len(precision_at_k_list) == 0:
            return 0

        # Average over all the precision at k for which the kth context is relevant.
        result = sum(precision_at_k_list) / len(precision_at_k_list)

        return Metric.context_precision(value=result)

    @llm_guided
    def compute_context_recall(
        self,
        query: Query,
    ) -> Metric:
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
        if len(query.prediction.context) == 0:
            raise ValueError(
                "Context recall requires context in the prediction response."
            )
        if len(query.groundtruth) == 0:
            raise ValueError("Context recall requires ground truth contexts.")

        scores = []
        for groundtruth in query.groundtruth:
            groundtruth_statements = self._generate_statements(
                text=groundtruth
            )

            verdicts = self._generate_context_recall_verdicts(
                context_list=query.prediction.context,
                groundtruth_statements=groundtruth_statements,
            )

            scores.append(
                sum(1 for verdict in verdicts if verdict["verdict"] == "yes")
                / len(verdicts)
            )

        result = max(scores)

        return Metric.context_recall(value=result)

    @llm_guided
    def compute_context_relevance(
        self,
        query: Query,
    ) -> Metric:
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
        if len(query.prediction.context) == 0:
            raise ValueError(
                "Context relevance requires context in the prediction response."
            )

        verdicts = self._generate_context_relevance_verdicts(
            query=query,
            context_list=query.prediction.context,
        )

        result = sum(
            1 for verdict in verdicts if verdict["verdict"] == "yes"
        ) / len(verdicts)

        return Metric.context_relevance(value=result)

    @llm_guided
    def compute_faithfulness(
        self,
        query: Query,
    ) -> Metric:
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
        if len(query.prediction.context) == 0:
            raise ValueError(
                "Faithfulness requires context in the prediction response."
            )

        claims = self._generate_claims(text=query.prediction.output)

        # If there aren't any claims, then the text is perfectly faithful, as the text does not contain any non-faithful claims.
        if len(claims) == 0:
            return 1

        faithfulness_verdicts = self._generate_faithfulness_verdicts(
            claims=claims,
            context_list=query.prediction.context,
        )

        result = sum(
            1
            for verdict in faithfulness_verdicts
            if verdict["verdict"] == "yes"
        ) / len(faithfulness_verdicts)

        return Metric.faithfulness(value=result)

    @llm_guided
    def compute_hallucination(
        self,
        query: Query,
    ) -> Metric:
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
        if len(query.prediction.context) == 0:
            raise ValueError(
                "Hallucination requires context in the prediction response."
            )

        verdicts = self._generate_hallucination_verdicts(
            text=query.prediction.output,
            context_list=query.prediction.context,
        )

        result = sum(
            1 for verdict in verdicts if verdict["verdict"] == "yes"
        ) / len(verdicts)

        return Metric.hallucination(value=result)

    @llm_guided
    def compute_summary_coherence(
        self,
        query: Query,
    ) -> Metric:
        """
        Compute summary coherence, the collective quality of a summary.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.

        Returns
        -------
        Metric
            The summary coherence score between 1 and 5. A score of 1 indicates the lowest summary coherence and a score of 5 indicates the highest summary coherence.
        """
        result = self._summary_coherence(
            text=query.query,
            summary=query.prediction.output,
        )
        return Metric.summary_coherence(value=result)

    @llm_guided
    def compute_toxicity(
        self,
        query: Query,
    ) -> Metric:
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
        opinions = self._generate_opinions(text=query.prediction.output)
        if len(opinions) == 0:
            return 0.0

        verdicts = self._generate_toxicity_verdicts(opinions=opinions)

        result = sum(
            1 for verdict in verdicts if verdict["verdict"] == "yes"
        ) / len(verdicts)

        return Metric.toxicity(value=result)

    def compute_rouge(
        self,
        query: Query,
        rouge_types: list[ROUGEType],
        use_stemmer: bool = False,
    ) -> list[Metric]:
        """
        Calculate ROUGE scores for a prediction given some set of references.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.
        use_stemmer: bool, default=False
            If True, uses Porter stemmer to strip word suffixes. Defaults to False.

        Returns
        -------
        list[Metric]
        """
        results = calculate_rouge_scores(
            predictions=[query.prediction.output],
            references=[query.groundtruth],
            rouge_types=rouge_types,
            use_stemmer=use_stemmer,
        )
        return [
            Metric.rouge(
                value=result,
                rouge_type=rouge_type,
            )
            for rouge_type, result in results.items()
        ]

    def compute_sentence_bleu(
        self,
        query: Query,
        weights: list[float] = [0.25, 0.25, 0.25, 0.25],
    ):
        """
        Calculate sentence BLEU scores for a set of prediction - ground truth pairs.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.
        weights: list[float], default=[0.25, 0.25, 0.25, 0.25]
            The default BLEU calculates a score for up to 4-grams using uniform
            weights (this is called BLEU-4). To evaluate your translations with
            higher/lower order ngrams, use customized weights. Example: when accounting
            for up to 5-grams with uniform weights (this is called BLEU-5) use [1/5]*5
        """
        result = calculate_sentence_bleu(
            predictions=query.prediction.output,
            references=query.groundtruth,
            weights=weights,
        )
        return Metric.bleu(
            value=result,
            weights=weights,
        )
