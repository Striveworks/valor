class EmptyEvaluatorException(Exception):
    def __init__(self):
        super().__init__(
            "evaluator cannot be finalized as it contains no data"
        )


class EmptyFilterException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class InternalCacheException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
