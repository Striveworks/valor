from valor_lite.text_generation.llm.exceptions import InvalidLLMResponseError


def validate_statements(
    response: dict[str, list[dict[str, str]]],
    key: str,
    allowed_values: set[str] | None = None,
    enforce_length: int | None = None,
):
    if key not in response:
        raise InvalidLLMResponseError(
            f"LLM did not include key '{key}' in its response: {response}"
        )
    elif (
        not isinstance(key, str)
        or not isinstance(response[key], list)
        or not all([isinstance(v, str) for v in response[key]])
    ):
        raise InvalidLLMResponseError(
            f"LLM response should follow the format 'dict[str, list[str]': {response}"
        )
    elif allowed_values is not None and not all(
        [v in allowed_values for v in response[key]]
    ):
        raise InvalidLLMResponseError(
            f"LLM response contains values from outside the allowed set {allowed_values}: {response}"
        )
    elif enforce_length is not None and enforce_length != len(response[key]):
        raise InvalidLLMResponseError(
            f"LLM response does not match input size of '{enforce_length}': {response}"
        )


def validate_verdicts(
    response: dict[str, list[dict[str, str]]],
    key: str,
    allowed_values: set[str] | None = None,
    enforce_length: int | None = None,
):
    if key not in response:
        raise InvalidLLMResponseError(
            f"LLM did not include key '{key}' in its response: {response}"
        )
    elif not isinstance(key, str) or not isinstance(response[key], list):
        raise InvalidLLMResponseError(
            f"LLM response should follow the format 'dict[str, list[dict[str, str]]]': {response}"
        )
    elif enforce_length is not None and enforce_length != len(response[key]):
        raise InvalidLLMResponseError(
            f"LLM response does not match input size of '{enforce_length}': {response}"
        )

    for value in response[key]:
        if not isinstance(value, dict):
            raise InvalidLLMResponseError(
                f"LLM response should follow the format 'dict[str, list[dict[str, str]]]': {response}"
            )
        elif set(value.keys()) != {"verdict", "analysis"}:
            raise InvalidLLMResponseError(
                f"LLM response is malformed. Inner dictionaries should only contain keys 'verdict' and 'analysis': {response} "
            )
        elif (
            allowed_values is not None
            and value["verdict"] not in allowed_values
        ):
            raise InvalidLLMResponseError(
                f"LLM response contains verdicts from outside the allowed set {allowed_values}: {response}"
            )
