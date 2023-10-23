from sqlalchemy import Select, and_, select
from sqlalchemy.sql.elements import BinaryExpression

from velour_api import schemas
from velour_api.backend import models


class Node:
    """Graph node representing a sql table.

    Contains all the information required to map a relationship between tables.

    Attributes
    ----------
    name : str
        The name of the node, used as a key in dictionaries.
    tablename : str
        The name of the sql table associated with this node.
    schema : Pydantic Schema
        Stores the pydantic object used to describe a row in the associated table.
    model:
        Stores the sqlalchemy object used to describe a row in the associated table.
    relationships : dict[Node, BinaryExpression]
        Maps the relationship to other nodes in join expressions.
    """

    def __init__(
        self,
        name: str,
        schema,
        model,
        tablename: str | None = None,
        relationships: dict["Node", BinaryExpression] | None = None,
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


class Graph:
    """
    Graph generator object.

    Attributes
    ----------
    dataset : Node
    model : Node
    datum : Node
    groundtruth : Node
    groundtruth_annotation : Node
    groundtruth_labels : Node
    prediction : Node
    prediction_annotation : Node
    prediction_labels : Node
    """

    def __init__(self):
        # create nodes
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
        self.groundtruth_annotation = Node(
            "groundtruth_annotation",
            tablename="annotation",
            schema=schemas.Annotation,
            model=models.Annotation,
        )
        self.prediction_annotation = Node(
            "prediction_annotation",
            tablename="annotation",
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
            tablename="label",
            schema=schemas.Label,
            model=models.Label,
        )
        self.prediction_label = Node(
            "prediction_label",
            tablename="label",
            schema=schemas.Label,
            model=models.Label,
        )

        # connect nodes
        self.dataset.connect(
            self.datum, models.Dataset.id == models.Datum.dataset_id
        )
        self.model.connect(
            self.prediction_annotation,
            models.Model.id == models.Annotation.model_id,
        )
        self.datum.connect(
            self.dataset, models.Datum.dataset_id == models.Dataset.id
        )
        self.datum.connect(
            self.groundtruth_annotation,
            models.Datum.id == models.Annotation.datum_id,
        )
        self.datum.connect(
            self.prediction_annotation,
            models.Datum.id == models.Annotation.datum_id,
        )
        self.groundtruth_annotation.connect(
            self.datum, models.Annotation.datum_id == models.Datum.id
        )
        self.groundtruth_annotation.connect(
            self.groundtruth,
            models.Annotation.id == models.GroundTruth.annotation_id,
        )
        self.prediction_annotation.connect(
            self.datum, models.Annotation.datum_id == models.Datum.id
        )
        self.prediction_annotation.connect(
            self.model, models.Annotation.model_id == models.Model.id
        )
        self.prediction_annotation.connect(
            self.prediction,
            models.Annotation.id == models.Prediction.annotation_id,
        )
        self.groundtruth.connect(
            self.groundtruth_annotation,
            models.GroundTruth.annotation_id == models.Annotation.id,
        )
        self.groundtruth.connect(
            self.groundtruth_label,
            models.GroundTruth.label_id == models.Label.id,
        )
        self.prediction.connect(
            self.prediction_annotation,
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

    def _walk_graph(self, root: Node, leaves: list[Node]) -> list[list[Node]]:
        """Walk the graph for each root to leaf pairing."""
        walks = []
        for leaf in leaves:
            if leaf is None or leaf == root:
                continue
            walk = _recursive_acyclic_walk(root, leaf, list(), set())
            walks.extend(walk)
        return walks

    def _reduce(self, walks: list[list[Node]]) -> list[Node]:
        """
        Flatten result of `Graph._walk_graph` into a sequence that respects a causal order-of-operations and
        has no repeating nodes.

        Parameters
        ----------
        walks : list[list[Node]]
            List of multiple graph walks that solve root to leaf connections.
        """
        visited = set()
        sequence = []
        for walk in walks:
            for node in walk:
                if node not in visited:
                    visited.add(node)
                    sequence.append(node)
        return sequence

    def _prune_relationships(
        self,
        sequence: list[Node],
    ) -> list[Node]:
        """
        Prune unused nodal relationships.

        This function should only be called after the `Graph._reduce` member function. The result of `_reduce` is
        a causal sequence of nodes. This means we can iterate through the sequence of nodes and remove relationships
        for nodes that have no causal relationship. This significantly cleans up the generated sql statement as
        unneccessary join relations are removed.
        """
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

    def _generate_query(
        self,
        target_node: Node,
        filter_nodes: list[Node],
        filters: list[BinaryExpression],
    ) -> Select:
        """
        Constructs a sql query.

        Generates the minimum join sequence that is required to access all the the filtering nodes from
        the target node.
        """

        # generate minimum sequence of table relationships
        graph = self._walk_graph(target_node, filter_nodes)
        sequence = self._reduce(graph)
        minseq = self._prune_relationships(sequence)

        # genrate sqlalchemy statement
        query = select(target_node.model.id)
        for node in minseq[1:]:
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

    def generate_query(
        self, target: Node, filters: dict[Node, list[BinaryExpression]]
    ) -> Select:
        """
        Generates a sqlalchemy select statement.

        The output statement automatically generates the joins required to
        evaluate the filters. The resulting statement can be read as:

        `SELECT target.id FROM target JOIN ... WHERE filters`

        Parameters
        ----------
        target : Node
            Node that is the target of sql `SELECT` statement.
        filters : dict[Node, list[BinaryExpression]]
            Dictionary of filtering expressions stored by the node that is required to evaluate them.

        Returns
        -------
        sqlalchemy.Select


        Examples
        -------
        Generate a query of labels that is constrained to groundtruths from a dataset with name `dset`.
        >>> graph = Graph()
        >>> target = graph.groundtruth_label
        >>> filter1 = [models.Dataset.name == "dset"]
        >>> filter2 = [models.Annotation.model_id.is_(None)]
        >>> filters = {graph.Dataset: filter1, graph.Annotation: filter2}
        >>> q = graph.generate_query(graph.dataset, filters)
        """

        if target is None:
            raise ValueError("Target node cannot be `NoneType`")

        filter_nodes = set(filters.keys())
        gt_only_set = {
            self.groundtruth_label,
            self.groundtruth,
            self.groundtruth_annotation,
        }
        pd_only_set = {
            self.prediction_label,
            self.prediction,
            self.prediction_annotation,
            self.model,
        }

        subtarget = None
        filter_subnodes = None
        if target in gt_only_set and pd_only_set.intersection(filter_nodes):
            # groundtruth target requires groundtruth filter_nodes
            subtarget = self.datum
            filter_subnodes = pd_only_set.intersection(filter_nodes)
            filter_nodes = filter_nodes - pd_only_set.intersection(
                filter_nodes
            )
            filter_nodes.add(self.datum)
        elif target in pd_only_set and gt_only_set.intersection(filter_nodes):
            # prediction target requires groundtruth filter_nodes
            subtarget = self.datum
            filter_subnodes = gt_only_set.intersection(filter_nodes)
            filter_nodes = filter_nodes - gt_only_set.intersection(
                filter_nodes
            )
            filter_nodes.add(self.datum)
        elif (
            target not in gt_only_set.union(pd_only_set)
            and gt_only_set.intersection(filter_nodes)
            and pd_only_set.intersection(filter_nodes)
        ):
            # target is neither groundtruth or prediction specific but filters are from both.
            subtarget = self.datum
            filter_subnodes = pd_only_set.intersection(filter_nodes)
            filter_nodes = filter_nodes - pd_only_set.intersection(
                filter_nodes
            )
            filter_nodes.add(self.datum)

        # create sqlalchemy query
        query = self._generate_query(target, filter_nodes, filters)
        if subtarget and filter_subnodes:
            subquery = self._generate_query(
                subtarget, filter_subnodes, filters
            )
            query = query.where(models.Datum.id.in_(subquery))

        return query

    def select_groundtruth_graph_node(self, table):
        """Takes models.<table> and returns associated Node."""
        match table:
            case models.Dataset:
                return self.dataset
            case models.Model:
                return self.model
            case models.Datum:
                return self.datum
            case models.Annotation:
                return self.groundtruth_annotation
            case models.GroundTruth:
                return self.groundtruth
            case models.Prediction:
                return self.prediction
            case models.Label:
                return self.groundtruth_label

    def select_prediction_graph_node(self, table):
        """Takes models.<table> and returns associated Node."""
        match table:
            case models.Dataset:
                return self.dataset
            case models.Model:
                return self.model
            case models.Datum:
                return self.datum
            case models.Annotation:
                return self.prediction_annotation
            case models.GroundTruth:
                return self.groundtruth
            case models.Prediction:
                return self.prediction
            case models.Label:
                return self.prediction_label
