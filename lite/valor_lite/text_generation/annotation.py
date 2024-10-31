from dataclasses import dataclass, field


@dataclass
class Context:
    """
    Contextual ground truth and prediction.

    Attributes
    ----------
    groundtruth : list[str]
        The definitive context.
    prediction : list[str]
        Any retrieved context from a retrieval-augmented-generation (RAG) pipeline.

    Examples
    --------
    ... context = Context(
    ...     groundtruth=[...],
    ...     prediction=[...],
    ... )
    """

    groundtruth: list[str] = field(default_factory=list)
    prediction: list[str] = field(default_factory=list)


@dataclass
class QueryResponse:
    """
    Text generation data structure containing ground truths and predictions.

    Attributes
    ----------
    query : str
        The user query.
    response : str
        The language model's response.
    context : Context
        Any context provided to the model.

    Examples
    --------
    >>> query = QueryResponse(
    ...     query='When was George Washington born?',
    ...     response="February 22, 1732",
    ...     context=Context(
    ...         groundtruth=["02/22/1732"],
    ...         prediction=["02/22/1732"],
    ...     ),
    ... )
    """

    query: str
    response: str
    context: Context | None = field(default=None)
