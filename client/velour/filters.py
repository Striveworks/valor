class DeclarativeMapper:
    def __init__(self, object_type: object):
        self.object_type = object_type
        self.expressions = []

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, self.object_type):
            raise ValueError(f"Value should be of type `{self.object_type}`")
        self.expressions.append(("==", __value))

    def __ne__(self, __value: object) -> bool:
        if not isinstance(__value, self.object_type):
            raise ValueError(f"Value should be of type `{self.object_type}`")
        self.expressions.append(("!=", __value))

    def __lt__(self, __value: object) -> bool:
        if isinstance(__value, str):
            raise TypeError("`__lt__` does not support type `str`")
        if not isinstance(__value, self.object_type):
            raise ValueError(f"Value should be of type `{self.object_type}`")
        self.expressions.append(("<", __value))

    def __gt__(self, __value: object) -> bool:
        if isinstance(__value, str):
            raise TypeError("`__gt__` does not support type `str`")
        if not isinstance(__value, self.object_type):
            raise ValueError(f"Value should be of type `{self.object_type}`")
        self.expressions.append((">", __value))

    def __le__(self, __value: object) -> bool:
        if isinstance(__value, str):
            raise TypeError("`__le__` does not support type `str`")
        if not isinstance(__value, self.object_type):
            raise ValueError(f"Value should be of type `{self.object_type}`")
        self.expressions.append(("<=", __value))

    def __ge__(self, __value: object) -> bool:
        if isinstance(__value, str):
            raise TypeError("`__ge__` does not support type `str`")
        if not isinstance(__value, self.object_type):
            raise ValueError(f"Value should be of type `{self.object_type}`")
        self.expressions.append((">=", __value))


class Metadata:
    key = DeclarativeMapper(str)
    value = DeclarativeMapper(str)


class Dataset:
    id = DeclarativeMapper(int)
    name = DeclarativeMapper(str)
    metadata = Metadata()


class Model:
    pass


class Datum:
    pass


class Annotation:
    pass


d = Dataset()
d.id == 1

print(d.id.expressions)
