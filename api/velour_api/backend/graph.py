from sqlalchemy.sql.elements import BinaryExpression

from velour_api import schemas
from velour_api.backend import models


class Node:
    def __init__(self, name: str, schema, model, edge: bool = False):
        self.name = name
        self.schema = schema
        self.model = model
        self.relationships = {}
        self.edge = edge

    def __contains__(self, node: "Node"):
        if not isinstance(node, Node):
            raise TypeError("expected `Node` type.")
        return node in self.relationships

    def __eq__(self, other):
        if not isinstance(other, Node):
            return self.name == other
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def connect(self, node: "Node", relationship: BinaryExpression):
        if not isinstance(node, Node):
            raise TypeError("expected `Node` type.")
        self.relationships[node] = relationship


class DirectedAcyclicGraph:
    def __init__(self, root: Node):
        self.root = root
        self.nodes = set(root.relationships.keys())

    def _walk_recursion(
        self, current: Node, target: Node, path: list[Node], invalid: set[Node]
    ):
        if current == target:
            path.append(current)
            return [path]
        elif current.edge:
            return None
        else:
            path.append(current)
            invalid.add(current)
            steps = []
            for node in current.relationships.keys():
                if node not in invalid:
                    step = self._walk_recursion(
                        node, target, path.copy(), invalid.copy()
                    )
                    if step is not None:
                        steps.extend(step)
            return steps if steps else None

    def walk(self, targets: list[Node]):
        walks = []
        for target in targets:
            if target == self.root:
                continue
            walks.extend(
                self._walk_recursion(self.root, target, list(), set())
            )
        return walks


class VelourDAG:
    def __init__(self):
        self.dataset = Node(
            "dataset",
            schema=schemas.Dataset,
            model=models.Dataset,
        )

        self.model = Node(
            "model",
            schema=schemas.Model,
            model=models.Model,
        )

        self.datum = Node(
            "datum",
            schema=schemas.Datum,
            model=models.Datum,
        )

        self.annotation = Node(
            "annotation",
            schema=schemas.Annotation,
            model=models.Annotation,
        )

        self.groundtruth = Node(
            "groundtruth",
            schema=schemas.GroundTruth,
            model=models.GroundTruth,
        )

        self.prediction = Node(
            "prediction",
            schema=schemas.Prediction,
            model=models.Prediction,
        )

        self.groundtruth_label = Node(
            "groundtruth_label",
            schema=schemas.Label,
            model=models.Label,
            edge=True,
        )

        self.prediction_label = Node(
            "prediction_label",
            schema=schemas.Label,
            model=models.Label,
            edge=True,
        )

        self.dataset_metadatum = Node(
            "dataset_metadatum",
            schema=schemas.MetaDatum,
            model=models.MetaDatum,
            edge=True,
        )

        self.model_metadatum = Node(
            "model_metadatum",
            schema=schemas.MetaDatum,
            model=models.MetaDatum,
            edge=True,
        )

        self.datum_metadatum = Node(
            "datum_metadatum",
            schema=schemas.MetaDatum,
            model=models.MetaDatum,
            edge=True,
        )

        self.annotation_metadatum = Node(
            "annotation_metadatum",
            schema=schemas.MetaDatum,
            model=models.MetaDatum,
            edge=True,
        )

        self.dataset.connect(
            self.datum, models.Dataset.id == models.Datum.dataset_id
        )
        self.dataset.connect(
            self.dataset_metadatum,
            models.Dataset.id == models.MetaDatum.dataset_id,
        )

        self.model.connect(
            self.annotation, models.Model.id == models.Annotation.model_id
        )
        self.model.connect(
            self.model_metadatum, models.Model.id == models.MetaDatum.model_id
        )

        self.datum.connect(
            self.dataset, models.Datum.dataset_id == models.Dataset.id
        )
        self.datum.connect(
            self.annotation, models.Datum.id == models.Annotation.datum_id
        )
        self.datum.connect(
            self.datum_metadatum, models.Datum.id == models.MetaDatum.datum_id
        )

        self.annotation.connect(
            self.datum, models.Annotation.datum_id == models.Datum.id
        )
        self.annotation.connect(
            self.model, models.Annotation.model_id == models.Model.id
        )
        self.annotation.connect(
            self.groundtruth,
            models.Annotation.id == models.GroundTruth.annotation_id,
        )
        self.annotation.connect(
            self.prediction,
            models.Annotation.id == models.Prediction.annotation_id,
        )
        self.annotation.connect(
            self.annotation_metadatum,
            models.Annotation.id == models.MetaDatum.annotation_id,
        )

        self.groundtruth.connect(
            self.annotation,
            models.GroundTruth.annotation_id == models.Annotation.id,
        )
        self.groundtruth.connect(
            self.groundtruth_label,
            models.GroundTruth.label_id == models.Label.id,
        )

        self.prediction.connect(
            self.annotation,
            models.Prediction.annotation_id == models.Annotation.id,
        )
        self.prediction.connect(
            self.prediction_label,
            models.Prediction.label_id == models.Label.id,
        )

        self.groundtruth_label.connect(
            self.groundtruth, models.Label.id == models.GroundTruth.label_id
        )
        self.prediction_label.connect(
            self.prediction, models.Label.id == models.Prediction.label_id
        )

        self.dataset_metadatum.connect(
            self.dataset, models.MetaDatum.dataset_id == models.Dataset.id
        )
        self.model_metadatum.connect(
            self.model, models.MetaDatum.model_id == models.Model.id
        )
        self.datum_metadatum.connect(
            self.datum, models.MetaDatum.datum_id == models.Datum.id
        )
        self.annotation_metadatum.connect(
            self.annotation,
            models.MetaDatum.annotation_id == models.Annotation.id,
        )

    def sequence(self, walks: list[list[Node]]):
        visited = set()
        sequence = []
        for walk in walks:
            for node in walk:
                if node not in visited:
                    visited.add(node)
                    sequence.append(node)
        return sequence

    def graph(self, root: Node, targets: list[Node]):
        g = DirectedAcyclicGraph(root)
        walks = g.walk(targets)
        return self.sequence(walks)


if __name__ == "__main__":

    g = VelourDAG()
