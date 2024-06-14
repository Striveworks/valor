from valor import (
    Annotation,
    Client,
    Dataset,
    Datum,
    Filter,
    GroundTruth,
    Label,
)
from valor.schemas import And, Box


def test_example_boats_and_swimmers(client: Client):

    contains_boat_swimmer = (
        ("uid1", False, False),
        ("uid2", True, False),
        ("uid3", False, True),
        ("uid4", True, True),
    )

    box = Box.from_extrema(0, 10, 0, 10)
    swimmer = Label(key="class", value="swimmer")
    boat = Label(key="class", value="boat")
    fish = Label(key="class", value="fish")

    dataset = Dataset.create("ocean_images")
    for uid, is_boat, is_swimmer in contains_boat_swimmer:
        dataset.add_groundtruth(
            GroundTruth(
                datum=Datum(uid=uid),
                annotations=[
                    Annotation(
                        labels=[boat if is_boat else fish],
                        bounding_box=box,
                        is_instance=True,
                    ),
                    Annotation(
                        labels=[swimmer if is_swimmer else fish],
                        bounding_box=box,
                        is_instance=True,
                    ),
                ],
            )
        )

    # Just fish
    just_fish = client.get_datums(
        Filter(
            datums=And(
                Label.key == "class",
                Label.value != "boat",
                Label.value != "swimmer",
            ),
        )
    )
    assert len(just_fish) == 1
    assert just_fish[0].uid == "uid1"

    # No swimmers
    no_swimmers = client.get_datums(
        Filter(
            datums=And(
                Label.key == "class",
                Label.value == "boat",
                Label.value != "swimmer",
            ),
        )
    )
    assert len(no_swimmers) == 1
    assert no_swimmers[0].uid == "uid2"

    # No boats
    no_boats = client.get_datums(
        Filter(
            datums=And(
                Label.key == "class",
                Label.value != "boat",
                Label.value == "swimmer",
            ),
        )
    )
    assert len(no_boats) == 1
    assert no_boats[0].uid == "uid3"

    # Both swimmers and boats
    swimmers_and_boats = client.get_datums(
        Filter(
            datums=And(
                Label.key == "class",
                Label.value == "boat",
                Label.value == "swimmer",
            ),
        )
    )
    assert len(swimmers_and_boats) == 1
    assert swimmers_and_boats[0].uid == "uid4"


def test_example_boats_of_different_sizes(client: Client):

    contains_boat_swimmer = (
        ("uid1", False, False),
        ("uid2", True, False),
        ("uid3", False, True),
        ("uid4", True, True),
    )

    small_box = Box.from_extrema(0, 5, 0, 5)
    large_box = Box.from_extrema(0, 10, 0, 10)

    swimmer = Label(key="class", value="swimmer")
    boat = Label(key="class", value="boat")
    fish = Label(key="class", value="fish")

    dataset = Dataset.create("ocean_images")
    for uid, is_large_boat, is_swimmer in contains_boat_swimmer:
        dataset.add_groundtruth(
            GroundTruth(
                datum=Datum(uid=uid),
                annotations=[
                    Annotation(
                        labels=[boat],
                        bounding_box=large_box if is_large_boat else small_box,
                        is_instance=True,
                    ),
                    Annotation(
                        labels=[swimmer if is_swimmer else fish],
                        bounding_box=small_box,
                        is_instance=True,
                    ),
                ],
            )
        )

    # No swimmer, small boats
    no_swimmer_small_boats = client.get_datums(
        Filter(
            datums=And(
                Label.key == "class",
                Label.value != "swimmer",
            ),
            annotations=And(
                Label.key == "class",
                Label.value == "boat",
                Annotation.bounding_box.area < 50,
            ),
        )
    )
    assert len(no_swimmer_small_boats) == 1
    assert no_swimmer_small_boats[0].uid == "uid1"

    # No swimmer, large boats
    no_swimmer_large_boats = client.get_datums(
        Filter(
            datums=And(
                Label.key == "class",
                Label.value != "swimmer",
            ),
            annotations=And(
                Label.key == "class",
                Label.value == "boat",
                Annotation.bounding_box.area > 50,
            ),
        )
    )
    assert len(no_swimmer_large_boats) == 1
    assert no_swimmer_large_boats[0].uid == "uid2"

    # Swimmer with small boat
    swimmer_with_small_boats = client.get_datums(
        Filter(
            datums=And(
                Label.key == "class",
                Label.value == "swimmer",
            ),
            annotations=And(
                Label.key == "class",
                Label.value == "boat",
                Annotation.bounding_box.area < 50,
            ),
        )
    )
    assert len(swimmer_with_small_boats) == 1
    assert swimmer_with_small_boats[0].uid == "uid3"

    # Swimmer with large boat
    swimmers_and_boats = client.get_datums(
        Filter(
            datums=And(
                Label.key == "class",
                Label.value == "swimmer",
            ),
            annotations=And(
                Label.key == "class",
                Label.value == "boat",
                Annotation.bounding_box.area > 50,
            ),
        )
    )
    assert len(swimmers_and_boats) == 1
    assert swimmers_and_boats[0].uid == "uid4"
