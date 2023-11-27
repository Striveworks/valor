def _split_query_params(param_string: str | None) -> list[str] | None:
    """Split GET query parameters and return a list when possible."""
    if not param_string:
        return None
    elif "," in param_string:
        return param_string.split(",")
    else:
        return [param_string]
