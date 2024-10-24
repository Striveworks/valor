from dataclasses import dataclass


@dataclass
class Response:
    """
    Response from a RAG pipeline.

    Attributes
    ----------
    output : str
        The model's response.
    context : list[str]
        Any retrieved context that the model was provided.
    """

    output: str
    context: list[str]


@dataclass
class Query:
    """
    Text generation data structure containing ground truths and predictions.

    Attributes
    ----------
    query : str
        The user query.
    groundtruths : list[str]
        A list of ground truth contexts.
    prediction : Response
        The response from the LLM.

    Examples
    --------
    >>> query = Query(
    ...     query='When was George Washington born?',
    ...     groundtruths=["02/22/1732"],
    ...     predictions=Response(
    ...         output="February 22, 1732",
    ...         context=[...],
    ...     ),
    ... )
    """

    query: str
    groundtruth: list[str]
    prediction: Response
