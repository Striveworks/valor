from dataclasses import dataclass

from valor import Datum


@dataclass
class Test:
    x: str


if __name__ == "__main__":
    print(Datum.uid.is_none())

    d = Datum(uid="uid1")

    print(d.uid.get_value())
