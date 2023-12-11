class SchemaTypeError(Exception):
    """
    Raises an exception if the user tries to pass the wrong schema as an argument

    Parameters
    -------
    member_name : str
        The name of the object.
    intended_type
        The expected type.
    value : object
        The value of the object.
    """

    def __init__(self, member_name: str, intended_type, value: object):
        return super().__init__(
            f"`{member_name}` should be of type `{str(intended_type)}`, received `{str(value)}`"
        )
