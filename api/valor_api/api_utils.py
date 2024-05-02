def _split_query_params(param_string: str | None) -> list[str] | None:
    """Split GET query parameters and return a list when possible."""
    if not param_string:
        return None
    elif "," in param_string:
        return param_string.split(",")
    else:
        return [param_string]


def _get_pagination_header(
    offset: int, number_of_returned_items: int, total_number_of_items: int
) -> dict[str, str]:
    """
    Returns the pagination header for use in our various GET endpoints.

    Parameters
    ----------
    offset : int
        The start index of the returned items.
    number_of_returned_items : int
        The number of items to be returned to the user.
    count : int
        The total number of items that could be returned to the user.

    Returns
    -------
    dict[str, str]
        The content-range header to attach to the response
    """

    if number_of_returned_items == 0:
        range_indicator = "*"
    else:
        end_index = (
            offset + number_of_returned_items - 1
        )  # subtract one to make it zero-indexed

        range_indicator = f"{offset}-{end_index}"

    return {
        "content-range": f"items {range_indicator}/{total_number_of_items}"
    }


def _validate_metrics_to_sort_by(metrics_to_sort_by: list[str] | None):
    """
    Checks that the user is only passing dataset-level metrics to metrics_to_sort_by.

    Parameters
    ----------
    metrics_to_sort_by: str, optional
        An optional list of metric types to sort evaluations by.

    Raises
    -------
    ValueError
        If metrics_to_sort_by contains one or more metrics that aren't at the dataset-level.
    """
    allowed_metrics = ["mAPAveragedOverIOUs", "mIOU", "mAccuracy"]

    if metrics_to_sort_by is not None:
        if not all(
            [metric in allowed_metrics for metric in metrics_to_sort_by]
        ):
            raise ValueError(
                f"metrics_to_sort_by contains metrics that are too granular to sort by. Please only pass the following metrics in metrics_to_sort_by: {allowed_metrics}"
            )
