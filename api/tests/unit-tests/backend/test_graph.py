import pytest

from velour_api.backend.graph import Graph, Node, _recursive_acyclic_walk


@pytest.fixture
def graph():
    return Graph()


@pytest.fixture
def model_to_dataset(graph):
    return [
        graph.model,
        graph.prediction_annotation,
        graph.datum,
        graph.dataset,
    ]


@pytest.fixture
def model_to_groundtruth_labels(graph):
    return [
        graph.model,
        graph.prediction_annotation,
        graph.datum,
        graph.groundtruth_annotation,
        graph.groundtruth,
        graph.groundtruth_label,
    ]


@pytest.fixture
def model_to_prediction_labels(graph):
    return [
        graph.model,
        graph.prediction_annotation,
        graph.prediction,
        graph.prediction_label,
    ]


@pytest.fixture
def dataset_to_model(model_to_dataset):
    return list(reversed(model_to_dataset))


@pytest.fixture
def dataset_to_groundtruth_labels(graph):
    return [
        graph.dataset,
        graph.datum,
        graph.groundtruth_annotation,
        graph.groundtruth,
        graph.groundtruth_label,
    ]


@pytest.fixture
def dataset_to_prediction_labels(graph):
    return [
        graph.dataset,
        graph.datum,
        graph.prediction_annotation,
        graph.prediction,
        graph.prediction_label,
    ]


@pytest.fixture
def groundtruth_labels_to_model(model_to_groundtruth_labels):
    return list(reversed(model_to_groundtruth_labels))


@pytest.fixture
def groundtruth_labels_to_dataset(dataset_to_groundtruth_labels):
    return list(reversed(dataset_to_groundtruth_labels))


@pytest.fixture
def groundtruth_labels_to_prediction_labels(graph):
    return [
        graph.groundtruth_label,
        graph.groundtruth,
        graph.groundtruth_annotation,
        graph.datum,
        graph.prediction_annotation,
        graph.prediction,
        graph.prediction_label,
    ]


@pytest.fixture
def prediction_labels_to_model(model_to_prediction_labels):
    return list(reversed(model_to_prediction_labels))


@pytest.fixture
def prediction_labels_to_dataset(dataset_to_prediction_labels):
    return list(reversed(dataset_to_prediction_labels))


@pytest.fixture
def prediction_labels_to_groundtruth_labels(
    groundtruth_labels_to_prediction_labels,
):
    return list(reversed(groundtruth_labels_to_prediction_labels))


def test__recursive_acyclic_walk(
    graph,
    model_to_dataset,
    model_to_groundtruth_labels,
    model_to_prediction_labels,
    dataset_to_model,
    dataset_to_groundtruth_labels,
    dataset_to_prediction_labels,
    groundtruth_labels_to_dataset,
    groundtruth_labels_to_model,
    groundtruth_labels_to_prediction_labels,
    prediction_labels_to_dataset,
    prediction_labels_to_model,
    prediction_labels_to_groundtruth_labels,
):
    # model --> extremities
    root = graph.model
    assert _recursive_acyclic_walk(root, graph.dataset, list(), set()) == [
        model_to_dataset
    ]
    assert _recursive_acyclic_walk(
        root, graph.groundtruth_label, list(), set()
    ) == [model_to_groundtruth_labels]
    assert _recursive_acyclic_walk(
        root, graph.prediction_label, list(), set()
    ) == [model_to_prediction_labels]

    # dataset --> extremities
    root = graph.dataset
    assert _recursive_acyclic_walk(root, graph.model, list(), set()) == [
        dataset_to_model
    ]
    assert _recursive_acyclic_walk(
        root, graph.groundtruth_label, list(), set()
    ) == [dataset_to_groundtruth_labels]
    assert _recursive_acyclic_walk(
        root, graph.prediction_label, list(), set()
    ) == [dataset_to_prediction_labels]

    # groundtruth labels --> extremities
    root = graph.groundtruth_label
    assert _recursive_acyclic_walk(root, graph.model, list(), set()) == [
        groundtruth_labels_to_model
    ]
    assert _recursive_acyclic_walk(root, graph.dataset, list(), set()) == [
        groundtruth_labels_to_dataset
    ]
    assert _recursive_acyclic_walk(
        root, graph.prediction_label, list(), set()
    ) == [groundtruth_labels_to_prediction_labels]

    # prediction labels --> extremities
    root = graph.prediction_label
    assert _recursive_acyclic_walk(root, graph.model, list(), set()) == [
        prediction_labels_to_model
    ]
    assert _recursive_acyclic_walk(root, graph.dataset, list(), set()) == [
        prediction_labels_to_dataset
    ]
    assert _recursive_acyclic_walk(
        root, graph.groundtruth_label, list(), set()
    ) == [prediction_labels_to_groundtruth_labels]


def test__walk_graph(
    graph,
    model_to_dataset,
    model_to_groundtruth_labels,
    model_to_prediction_labels,
    dataset_to_model,
    dataset_to_groundtruth_labels,
    dataset_to_prediction_labels,
    groundtruth_labels_to_dataset,
    groundtruth_labels_to_model,
    groundtruth_labels_to_prediction_labels,
    prediction_labels_to_dataset,
    prediction_labels_to_model,
    prediction_labels_to_groundtruth_labels,
):
    def leaves(root: Node):
        return [
            node
            for node in [
                graph.model,
                graph.dataset,
                graph.groundtruth_label,
                graph.prediction_label,
            ]
            if node != root
        ]

    root = graph.model
    assert graph._walk_graph(root, leaves(root)) == (
        [
            model_to_dataset,
            model_to_groundtruth_labels,
            model_to_prediction_labels,
        ]
    )

    root = graph.dataset
    assert graph._walk_graph(root, leaves(root)) == (
        [
            dataset_to_model,
            dataset_to_groundtruth_labels,
            dataset_to_prediction_labels,
        ]
    )

    root = graph.groundtruth_label
    assert graph._walk_graph(root, leaves(root)) == (
        [
            groundtruth_labels_to_model,
            groundtruth_labels_to_dataset,
            groundtruth_labels_to_prediction_labels,
        ]
    )

    root = graph.prediction_label
    assert graph._walk_graph(root, leaves(root)) == (
        [
            prediction_labels_to_model,
            prediction_labels_to_dataset,
            prediction_labels_to_groundtruth_labels,
        ]
    )


def test__reduce(graph):
    root = graph.model
    leaves = [graph.dataset, graph.groundtruth_label, graph.prediction_label]
    walks = graph._walk_graph(root, leaves)

    assert graph._reduce(walks) == [
        graph.model,
        graph.prediction_annotation,
        graph.datum,
        graph.dataset,
        graph.groundtruth_annotation,
        graph.groundtruth,
        graph.groundtruth_label,
        graph.prediction,
        graph.prediction_label,
    ]


def test__prune_relationships(graph):
    root = graph.model
    leaves = [graph.dataset, graph.groundtruth_label, graph.prediction_label]
    walks = graph._walk_graph(root, leaves)
    sequence = graph._reduce(walks)

    num_relationships_before = [1, 3, 3, 1, 2, 2, 1, 2, 1]
    for node, count in zip(sequence, num_relationships_before):
        assert len(node.relationships) == count

    pruned_sequence = graph._prune_relationships(sequence)

    num_relationships_after = [0, 1, 1, 1, 1, 1, 1, 1, 1]
    for node, count in zip(pruned_sequence, num_relationships_after):
        assert len(node.relationships) == count

    assert num_relationships_before != num_relationships_after
