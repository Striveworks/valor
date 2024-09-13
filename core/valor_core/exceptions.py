class InvalidLLMResponseError(Exception):
    """
    Raised when the response from the LLM is invalid for a given metric computation.
    """

    pass
