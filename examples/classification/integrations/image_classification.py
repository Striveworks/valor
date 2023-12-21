from typing import List, Dict

from velour import Dataset, Model, Datum, Annotation, GroundTruth, Prediction, Label


labels = [
    [
        Label(key="class", value="dog"),
        Label(key="superclass", value="animal")
    ],
    [
        Label(key="class", value="cat"),
        Label(key="superclass", value="animal")
    ],
    [
        Label(key="class", value="car"),
        Label(key="superclass", value="vehicle")
    ],
    [
        Label(key="class", value="truck"),
        Label(key="superclass", value="vehicle")
    ]
]


def get_labels(i: int) -> List[Label]:
    global labels
    return labels[i % 4]


def get_scored_labels(i: int) -> List[Label]:
    

