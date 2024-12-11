class InvalidLLMResponseError(Exception):
    """
    Raised when the response from the LLM is invalid for a given metric computation.
    """

    pass


class MismatchingTextGenerationDatumError(Exception):
    """
    Raised when datums with the same uid but different text are added to the ValorTextGenerationStreamingManager.
    """

    pass
