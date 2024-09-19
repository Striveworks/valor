try:
    import numpy as np  # noqa F401
except ModuleNotFoundError:
    raise RuntimeError(
        "Using `valor.metrics` requires additional dependencies."
        " Please install with `pip install valor-client[local-metrics]`."
    )
