class SchemaTypeError(Exception):
    def __init__(self, member_name: str, intended_type, value: object):
        return super().__init__(
            f"`{member_name}` should be of type `{str(intended_type)}`, received `{str(value)}`"
        )


class SchemaValueError(Exception):
    def __init__(self, member_name: str, intended_type, value: object):
        return super().__init__(
            f"`{member_name}` should be of type `{str(intended_type)}`, received `{str(value)}`"
        )
