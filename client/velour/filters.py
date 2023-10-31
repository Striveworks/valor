class DeclarativeMapper:
    def __init__(self, name: str, object_type: object):
        self.name = name
        self.object_type = object_type
        self.expressions = []

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, self.object_type):
            raise ValueError(
                f"`{self.name}` should be of type `{self.object_type}`"
            )
        return (self.name, "==", __value)

    def __ne__(self, __value: object) -> bool:
        if not isinstance(__value, self.object_type):
            raise ValueError(
                f"`{self.name}` should be of type `{self.object_type}`"
            )
        return (self.name, "!=", __value)

    def __lt__(self, __value: object) -> bool:
        if isinstance(__value, str):
            raise TypeError("`__lt__` does not support type `str`")
        if not isinstance(__value, self.object_type):
            raise ValueError(
                f"`{self.name}` should be of type `{self.object_type}`"
            )
        return (self.name, "<", __value)

    def __gt__(self, __value: object) -> bool:
        if isinstance(__value, str):
            raise TypeError("`__gt__` does not support type `str`")
        if not isinstance(__value, self.object_type):
            raise ValueError(
                f"`{self.name}` should be of type `{self.object_type}`"
            )
        return (self.name, ">", __value)

    def __le__(self, __value: object) -> bool:
        if isinstance(__value, str):
            raise TypeError("`__le__` does not support type `str`")
        if not isinstance(__value, self.object_type):
            raise ValueError(
                f"`{self.name}` should be of type `{self.object_type}`"
            )
        return (self.name, "<=", __value)

    def __ge__(self, __value: object) -> bool:
        if isinstance(__value, str):
            raise TypeError("`__ge__` does not support type `str`")
        if not isinstance(__value, self.object_type):
            raise ValueError(
                f"`{self.name}` should be of type `{self.object_type}`"
            )
        return (self.name, ">=", __value)

    def in_(self, __values: list[object]) -> bool:
        pass

    def is_(self, __value: object) -> bool:
        pass

    def is_not(self, __value: object) -> bool:
        pass


class Metadata:
    def __init__(self, name: str):
        self.name = name
        self.key = DeclarativeMapper(name + ".key", str)

    def __getitem__(self, key: str):
        return DeclarativeMapper(self.name + "." + key + ".value", object)


class Dataset:
    id = DeclarativeMapper("dataset.id", int)
    name = DeclarativeMapper("dataset.name", str)
    metadata = Metadata("dataset.metadata")


class Model:
    pass


class Datum:
    pass


class Annotation:
    pass


expressions = [
    Dataset.id == 1,
    Dataset.name == "hello",
    Dataset.metadata.key == "angle",
    Dataset.metadata["angle"] > 0.5,
]
print(expressions)
