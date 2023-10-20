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


def _recursive_acyclic_walk(
    root: Node,
    leaf: Node,
    path: list[Node],
    invalid: set[Node],
):
    """Recursively populates the path argument."""
    if root == leaf:
        path.append(root)
        return [path]
    else:
        path.append(root)
        invalid.add(root)
        steps = []
        for node in root.relationships.keys():
            if node not in invalid:
                step = _recursive_acyclic_walk(
                    node, leaf, path.copy(), invalid.copy()
                )
                if step is not None:
                    steps.extend(step)
        return steps if steps else None


def _walk_graph(root: Node, leaves: list[Node]) -> list[list[Node]]:
    walks = []
    for leaf in leaves:
        if leaf == root:
            continue
        walk = _recursive_acyclic_walk(root, leaf, list(), set())
        walks.extend(walk)
    return walks


def _reduce(walks: list[list[Node]]) -> list[Node]:
    """flatten into a sequence that respects order-of-operations"""
    visited = set()
    sequence = []
    for walk in walks:
        for node in walk:
            if node not in visited:
                visited.add(node)
                sequence.append(node)
    return sequence


def _prune_relationships(sequence: list[Node]) -> list[Node]:
    """prune unused relationships"""
    _pruned_nodes = []
    for node in sequence:
        _pruned_nodes.append(
            Node(
                name=node.name,
                tablename=node.tablename,
                schema=node.schema,
                model=node.model,
                relationships={
                    relation: node.relationships[relation]
                    for relation in node.relationships
                    if relation in _pruned_nodes
                },
            )
        )
    return _pruned_nodes


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

groundtruth_annotation = Node(
    "groundtruth_annotation",
    tablename="annotation",
    schema=schemas.Annotation,
    model=models.Annotation,
)

prediction_annotation = Node(
    "prediction_annotation",
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

groundtruth_label = Node(
    "groundtruth_label",
    tablename="label",
    schema=schemas.Label,
    model=models.Label,
)

prediction_label = Node(
    "prediction_label",
    tablename="label",
    schema=schemas.Label,
    model=models.Label,
)

dataset.connect(datum, models.Dataset.id == models.Datum.dataset_id)

model.connect(
    prediction_annotation, models.Model.id == models.Annotation.model_id
)

datum.connect(dataset, models.Datum.dataset_id == models.Dataset.id)
datum.connect(
    groundtruth_annotation, models.Datum.id == models.Annotation.datum_id
)
datum.connect(
    prediction_annotation, models.Datum.id == models.Annotation.datum_id
)

groundtruth_annotation.connect(
    datum, models.Annotation.datum_id == models.Datum.id
)
groundtruth_annotation.connect(
    groundtruth,
    models.Annotation.id == models.GroundTruth.annotation_id,
)

prediction_annotation.connect(
    datum, models.Annotation.datum_id == models.Datum.id
)
prediction_annotation.connect(
    model, models.Annotation.model_id == models.Model.id
)
prediction_annotation.connect(
    prediction,
    models.Annotation.id == models.Prediction.annotation_id,
)

groundtruth.connect(
    groundtruth_annotation,
    models.GroundTruth.annotation_id == models.Annotation.id,
)
groundtruth.connect(
    groundtruth_label,
    models.GroundTruth.label_id == models.Label.id,
)

prediction.connect(
    prediction_annotation,
    models.Prediction.annotation_id == models.Annotation.id,
)
prediction.connect(
    prediction_label,
    models.Prediction.label_id == models.Label.id,
)

groundtruth_label.connect(
    groundtruth, models.Label.id == models.GroundTruth.label_id
)

prediction_label.connect(
    prediction, models.Label.id == models.Prediction.label_id
)


class SQLGraph(NamedTuple):
    dataset: Node
    model: Node
    datum: Node
    annotation: Node
    groundtruth: Node
    prediction: Node
    groundtruth_label: Node
    prediction_label: Node


def _generate_query(target_node, filter_nodes, filters):
    """Constructs a sql query"""

    # construct sql query
    graph = _walk_graph(target_node, filter_nodes)
    sequence = _reduce(graph)
    minwalk = _prune_relationships(sequence)
    query = select(target_node.model.id)
    for node in minwalk[1:]:
        query = query.join(
            node.model,
            and_(*list(node.relationships.values())),
        )

    # construct where expression
    expr = []
    for node in filter_nodes:
        if node in filters:
            expr.extend(filters[node])
    query = query.where(and_(*expr))

    return query


# generate sql alchemy relationships
def generate_query(target: Node, filters: dict[Node, list[BinaryExpression]]):
    """Generates joins and optionally a subquery to construct sql statement."""

    filter_nodes = set(filters.keys())
    gt_only_set = {groundtruth_label, groundtruth, groundtruth_annotation}
    pd_only_set = {prediction_label, prediction, prediction_annotation, model}

    subtarget = None
    filter_subnodes = None
    if target in gt_only_set and pd_only_set.intersection(filter_nodes):
        # groundtruth target requires groundtruth filter_nodes
        subtarget = datum
        filter_subnodes = pd_only_set.intersection(filter_nodes)
        filter_nodes = filter_nodes - pd_only_set.intersection(filter_nodes)
        filter_nodes.add(datum)
    elif target in pd_only_set and gt_only_set.intersection(filter_nodes):
        # prediction target requires groundtruth filter_nodes
        subtarget = datum
        filter_subnodes = gt_only_set.intersection(filter_nodes)
        filter_nodes = filter_nodes - gt_only_set.intersection(filter_nodes)
        filter_nodes.add(datum)
    elif (
        target not in gt_only_set.union(pd_only_set)
        and gt_only_set.intersection(filter_nodes)
        and pd_only_set.intersection(filter_nodes)
    ):
        # target is neither groundtruth or prediction specific but filters are from both.
        subtarget = datum
        filter_subnodes = pd_only_set.intersection(filter_nodes)
        filter_nodes = filter_nodes - pd_only_set.intersection(filter_nodes)
        filter_nodes.add(datum)

    query = _generate_query(target, filter_nodes, filters)

    subquery = None
    if subtarget and filter_subnodes:
        subquery = _generate_query(subtarget, filter_subnodes, filters)
        query = query.where(models.Datum.id.in_(subquery))

    return query


if __name__ == "__main__":
    target = groundtruth_label
    filters = {
        model: [models.Model.name == "model1"],
        dataset: [models.Dataset.name == "dataset1"],
        prediction_label: [models.Label.key == "k1"],
        groundtruth_label: [models.Label.key == "k1"],
    }
    # filters = {groundtruth_annotation}

    query = generate_query(target, filters)

    print(query)
