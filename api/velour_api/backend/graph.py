from typing import NamedTuple

from sqlalchemy import and_, select
from sqlalchemy.sql.elements import BinaryExpression

from velour_api import schemas
from velour_api.backend import models


class Node:
    def __init__(
        self,
        name: str,
        schema,
        model,
        tablename: str | None = None,
        relationships: dict[BinaryExpression] | None = None,
    ):
        self.name = name
        self.tablename = tablename if tablename else name
        self.schema = schema
        self.model = model
        self.relationships = relationships if relationships else {}

    def __str__(self):
        return f"Node: `{self.name}`, Table: `{self.tablename}`, Relationships: {len(self.relationships)}"

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


def recursive_acyclic_walk(
    root: Node,
    target: Node,
    path: list[Node],
    invalid: set[Node],
):
    """Recursively populates the path argument."""
    if root == target:
        path.append(root)
        return [path]
    else:
        path.append(root)
        invalid.add(root)
        steps = []
        for node in root.relationships.keys():
            if node not in invalid:
                step = recursive_acyclic_walk(
                    node, target, path.copy(), invalid.copy()
                )
                if step is not None:
                    steps.extend(step)
        return steps if steps else None


def create_acyclic_graph(root: Node, targets: list[Node]):
    walks = []
    for target in targets:
        if target == root:
            continue
        walk = recursive_acyclic_walk(root, target, list(), set())
        walks.extend(walk)
    return walks


def reduce(walks: list[list[Node]]):
    """flatten into a sequence that respects order-of-operations"""
    visited = set()
    sequence = []
    for walk in walks:
        for node in walk:
            if node not in visited:
                visited.add(node)
                sequence.append(node)
    return sequence


def prune(sequence: list[Node]):
    """prune unused relationships"""
    pruned_nodes = []
    for node in sequence:
        pruned_nodes.append(
            Node(
                name=node.name,
                tablename=node.tablename,
                schema=node.schema,
                model=node.model,
                relationships={
                    relation: node.relationships[relation]
                    for relation in node.relationships
                    if relation in pruned_nodes
                },
            )
        )
    return pruned_nodes


# Velour SQL Graph


dataset = Node(
    "dataset",
    schema=schemas.Dataset,
    model=models.Dataset,
)

model = Node(
    "model",
    schema=schemas.Model,
    model=models.Model,
)

datum = Node(
    "datum",
    schema=schemas.Datum,
    model=models.Datum,
)

annotation_groundtruth = Node(
    "annotation_groundtruth",
    tablename="annotation",
    schema=schemas.Annotation,
    model=models.Annotation,
)

annotation_prediction = Node(
    "annotation_prediction",
    tablename="annotation",
    schema=schemas.Annotation,
    model=models.Annotation,
)

groundtruth = Node(
    "groundtruth",
    schema=schemas.GroundTruth,
    model=models.GroundTruth,
)

prediction = Node(
    "prediction",
    schema=schemas.Prediction,
    model=models.Prediction,
)

label_groundtruth = Node(
    "label_groundtruth",
    tablename="label",
    schema=schemas.Label,
    model=models.Label,
)

label_prediction = Node(
    "label_prediction",
    tablename="label",
    schema=schemas.Label,
    model=models.Label,
)

dataset.connect(datum, models.Dataset.id == models.Datum.dataset_id)

model.connect(
    annotation_prediction, models.Model.id == models.Annotation.model_id
)

datum.connect(dataset, models.Datum.dataset_id == models.Dataset.id)
datum.connect(
    annotation_groundtruth, models.Datum.id == models.Annotation.datum_id
)
datum.connect(
    annotation_prediction, models.Datum.id == models.Annotation.datum_id
)

annotation_groundtruth.connect(
    datum, models.Annotation.datum_id == models.Datum.id
)
annotation_groundtruth.connect(
    groundtruth,
    models.Annotation.id == models.GroundTruth.annotation_id,
)

annotation_prediction.connect(
    datum, models.Annotation.datum_id == models.Datum.id
)
annotation_prediction.connect(
    model, models.Annotation.model_id == models.Model.id
)
annotation_prediction.connect(
    prediction,
    models.Annotation.id == models.Prediction.annotation_id,
)

groundtruth.connect(
    annotation_groundtruth,
    models.GroundTruth.annotation_id == models.Annotation.id,
)
groundtruth.connect(
    label_groundtruth,
    models.GroundTruth.label_id == models.Label.id,
)

prediction.connect(
    annotation_prediction,
    models.Prediction.annotation_id == models.Annotation.id,
)
prediction.connect(
    label_prediction,
    models.Prediction.label_id == models.Label.id,
)

label_groundtruth.connect(
    groundtruth, models.Label.id == models.GroundTruth.label_id
)

label_prediction.connect(
    prediction, models.Label.id == models.Prediction.label_id
)


class SQLGraph(NamedTuple):
    dataset: Node
    model: Node
    datum: Node
    annotation: Node
    groundtruth: Node
    prediction: Node
    label_groundtruth: Node
    label_prediction: Node


# generate sql alchemy relationships
def generate_query(target: Node, filters: set[Node]):
    """Generates joins and optionally a subquery to construct sql statement."""

    gt_only_set = {label_groundtruth, groundtruth, annotation_groundtruth}
    pd_only_set = {label_prediction, prediction, annotation_prediction, model}

    subtarget = None
    subfilters = None
    if target in gt_only_set and pd_only_set.intersection(filters):
        # groundtruth target requires groundtruth filters
        subtarget = datum
        subfilters = pd_only_set.intersection(filters)
        filters = filters - pd_only_set.intersection(filters)
        filters.add(datum)
    elif target in pd_only_set and gt_only_set.intersection(filters):
        # prediction target requires groundtruth filters
        subtarget = datum
        subfilters = gt_only_set.intersection(filters)
        filters = filters - gt_only_set.intersection(filters)
        filters.add(datum)

    # construct sql query
    graph = create_acyclic_graph(target, filters)
    sequence = reduce(graph)
    minwalk = prune(sequence)
    query = select(target.model.id)
    for node in minwalk[1:]:
        query = query.join(
            node.model,
            and_(*list(node.relationships.values())),
        )

    # (edge case) construct sql subquery
    subquery = None
    if subtarget and subfilters:
        subgraphs = create_acyclic_graph(subtarget, subfilters)
        subseq = reduce(subgraphs)
        subminwalk = prune(subseq)

        # construct sql query
        subquery = select(subtarget.model.id)
        for node in subminwalk[1:]:
            subquery = subquery.join(
                node.model,
                and_(*list(node.relationships.values())),
            )

    return query, subquery


if __name__ == "__main__":
    target = label_groundtruth
    filters = {model, dataset, label_prediction, label_groundtruth}
    # filters = {annotation_groundtruth}

    query, subquery = generate_query(target, filters)

    print(query)
    print()
    print(subquery)
