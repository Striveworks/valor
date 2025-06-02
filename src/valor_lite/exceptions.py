class EmptyEvaluatorError(Exception):
    def __init__(self):
        super().__init__(
            "evaluator cannot be finalized as it contains no data"
        )


class EmptyFilterError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class InternalCacheError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
