class InvalidLLMResponseError(Exception):
    """
    Raised when the response from the LLM is invalid for a given metric computation.
    """

    pass


class MismatchingTextGenerationDataError(Exception):
    """
    Raised when datums with the same uid but different texts are added to the ValorTextGenerationStreamingManager.
    """

    pass
