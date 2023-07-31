import numpy as np
import pytest

from velour import enums, schemas
from velour.schemas.core import _validate_href


@pytest.fixture
def box_points() -> list[schemas.Point]:
    return [
        schemas.Point(x=0, y=0),
        schemas.Point(x=10, y=0),
        schemas.Point(x=10, y=10),
        schemas.Point(x=0, y=10),
    ]


@pytest.fixture
def basic_polygon(box_points) -> schemas.BasicPolygon:
    return schemas.BasicPolygon(points=box_points)


@pytest.fixture
def bbox() -> schemas.BoundingBox:
    return schemas.BoundingBox.from_extrema(xmin=0, xmax=10, ymin=0, ymax=10)


@pytest.fixture
def polygon(basic_polygon) -> schemas.Polygon:
    return schemas.Polygon(boundary=basic_polygon)


@pytest.fixture
def raster_raw_mask() -> np.ndarray:
    """
    Creates a 2d numpy of bools of shape:
    | T  F |
    | F  T |
    """
    ones = np.ones((10, 10))
    zeros = np.zeros((10, 10))
    top = np.concatenate((ones, zeros), axis=1)
    bottom = np.concatenate((zeros, ones), axis=1)
    return np.concatenate((top, bottom), axis=0) == 1


@pytest.fixture
def raster(raster_raw_mask) -> schemas.Raster:
    return schemas.Raster.from_numpy(raster_raw_mask)


@pytest.fixture
def metadatum() -> schemas.MetaDatum:
    return schemas.MetaDatum(
        key="test",
        value=1234,
    )


""" velour.schemas.metadata """


def test_metadata_geojson():
    # @TODO: Implement GeoJSON
    schemas.GeoJSON(type="this shouldnt work", coordinates=[])


""" velour.schemas.geometry """


def test_geometry_point():
    # valid
    p1 = schemas.Point(x=1, y=1)
    p2 = schemas.Point(x=1.0, y=1.0)
    p3 = schemas.Point(x=1.0, y=0.99)

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        schemas.Point(x="1", y=1)
    assert "should be `float` type" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Point(x=1, y="1")
    assert "should be `float` type" in str(e)

    # test member fn `__hash__`
    assert p1.__hash__() == p2.__hash__()
    assert p1.__hash__() != p3.__hash__()

    # test member fn `resize`
    p11 = p1.resize(
        og_img_h=10,
        og_img_w=10,
        new_img_h=100,
        new_img_w=100,
    )
    assert p11.x == p1.x * 10
    assert p11.y == p1.y * 10


def test_geometry_box():
    p1 = schemas.Point(x=0, y=0)
    p2 = schemas.Point(x=10, y=10)
    p3 = schemas.Point(x=10, y=-10)
    p4 = schemas.Point(x=-10, y=10)

    # valid
    schemas.Box(min=p1, max=p2)

    # test `__post_init__`
    with pytest.raises(ValueError) as e:
        schemas.Box(min=p2, max=p1)
    assert "xmin > xmax" in e
    with pytest.raises(ValueError) as e:
        schemas.Box(min=p1, max=p4)
    assert "xmin > xmax" in e
    with pytest.raises(ValueError) as e:
        schemas.Box(min=p1, max=p3)
    assert "ymin > ymax" in e


def test_geometry_basicpolygon():
    p1 = schemas.Point(x=0, y=0)
    p2 = schemas.Point(x=5, y=0)
    p3 = schemas.Point(x=10, y=5)
    p4 = schemas.Point(x=0, y=5)

    schemas.BasicPolygon(points=[p1, p2, p4])

    # Test __post_init__
    with pytest.raises(TypeError) as e:
        schemas.BasicPolygon(points=p1)
    assert "is not a list" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.BasicPolygon(points=[p1, p2, 4])
    assert "is not a `Point`"
    with pytest.raises(ValueError) as e:
        schemas.BasicPolygon(points=[p1, p2])
    assert "needs at least 3 unique points" in str(e)

    # Test member fn `xy_list`
    poly = schemas.BasicPolygon(points=[p1, p2, p3])
    assert poly.xy_list() == [p1, p2, p3]

    # Test properties
    assert poly.xmin == 0
    assert poly.xmax == 10
    assert poly.ymin == 0
    assert poly.ymax == 5

    # Test classmethod `from_box`
    cmin = schemas.Point(x=-1, y=-2)
    cmax = schemas.Point(x=10, y=11)
    poly = schemas.BasicPolygon.from_box(
        box=schemas.Box(
            min=cmin,
            max=cmax,
        )
    )
    assert poly.xy_list() == [
        schemas.Point(x=-1, y=-2),
        schemas.Point(x=-1, y=11),
        schemas.Point(x=10, y=11),
        schemas.Point(x=10, y=-2),
    ]


def test_geometry_polygon():
    p1 = schemas.Point(x=0, y=0)
    p2 = schemas.Point(x=5, y=0)
    p3 = schemas.Point(x=0, y=5)
    poly = schemas.BasicPolygon(points=[p1, p2, p3])

    # valid
    schemas.Polygon(boundary=poly)
    schemas.Polygon(boundary=poly, holes=[poly])

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        schemas.Polygon(boundary=123)
    assert "boundary should be of type `velour.schemas.BasicPolygon`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Polygon(boundary=poly, holes=123)
    assert "holes should be a list of `velour.schemas.BasicPolygon`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Polygon(boundary=poly, holes=[123])
    assert (
        "should contain elements of type `velour.schemas.BasicPolygon`"
        in str(e)
    )


def test_geometry_boundingbox():
    p1 = schemas.Point(x=0, y=0)
    p2 = schemas.Point(x=5, y=0)
    p3 = schemas.Point(x=5, y=5)
    p4 = schemas.Point(x=0, y=5)
    poly = schemas.BasicPolygon(points=[p1, p2, p3, p4])

    # test __post_init__
    schemas.BoundingBox(polygon=poly)
    with pytest.raises(TypeError) as e:
        schemas.BoundingBox(polygon=p1)
    assert "should be of type `velour.schemas.BasicPolygon`" in str(e)
    with pytest.raises(ValueError) as e:
        schemas.BoundingBox(polygon=schemas.BasicPolygon(points=[p1, p2, p3]))
    assert "should be made of a 4-point polygon" in str(e)

    # test classmethod `from_extrema`
    bbox = schemas.BoundingBox.from_extrema(xmin=-1, xmax=10, ymin=-2, ymax=11)
    assert bbox.polygon.xy_list() == [
        schemas.Point(x=-1, y=-2),
        schemas.Point(x=10, y=-2),
        schemas.Point(x=10, y=11),
        schemas.Point(x=-1, y=11),
    ]


def test_geometry_multipolygon():
    p1 = schemas.Point(x=0, y=0)
    p2 = schemas.Point(x=5, y=0)
    p3 = schemas.Point(x=5, y=5)
    p4 = schemas.Point(x=0, y=5)
    component_poly = schemas.BasicPolygon(points=[p1, p2, p3, p4])
    poly = schemas.Polygon(boundary=component_poly)

    # valid
    schemas.MultiPolygon(polygons=[poly])

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        schemas.MultiPolygon(polygons=component_poly)
    assert "polygons should be list of `velour.schemas.Polyon`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.MultiPolygon(polygons=[component_poly])
    assert (
        "polygons list should contain elements of type `velour.schemas.Polygon`"
        in str(e)
    )


def test_geometry_raster(raster_raw_mask):
    mask1 = np.ones((10, 10)) == 1

    # valid
    schemas.Raster(mask="test", height=0, width=0)
    schemas.Raster.from_numpy(mask=mask1)

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        schemas.Raster(mask=123, height=10, width=10)
    assert "mask should be of type `str`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Raster(mask="123", height=12)
    with pytest.raises(TypeError) as e:
        schemas.Raster(mask="123", height=(1, 2, 3), width=12)
    with pytest.raises(TypeError) as e:
        schemas.Raster(mask="123", height=12.4, width="test")

    # test classmethod `from_numpy`
    mask2 = np.ones((10, 10, 10)) == 1
    mask3 = np.ones((10, 10))
    with pytest.raises(ValueError) as e:
        schemas.Raster.from_numpy(mask2)
    assert "raster currently only supports 2d arrays" in str(e)
    with pytest.raises(ValueError) as e:
        schemas.Raster.from_numpy(mask3)
    assert "Expecting a binary mask" in str(e)

    # test member fn `to_numpy`
    r = schemas.Raster.from_numpy(raster_raw_mask)
    assert (
        r.mask
        == "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUAQAAAACl8iCgAAAAF0lEQVR4nGP4f4CBiYGBIGZgsP9AjDoAuysDE0GVDN8AAAAASUVORK5CYII="
    )
    assert r.height == 20
    assert r.width == 20
    assert (r.to_numpy() == raster_raw_mask).all()


""" velour.schemas.core """


def test_core__validate_href():
    _validate_href("http://test")
    _validate_href("https://test")

    with pytest.raises(ValueError) as e:
        _validate_href("test")
    assert "`href` must start with http:// or https://" in str(e)
    with pytest.raises(TypeError) as e:
        _validate_href(1)
    assert "passed something other than 'str'" in str(e)


def test_core_metadatum():
    schemas.MetaDatum(key="test", value="test")
    schemas.MetaDatum(key="test", value=1)
    schemas.MetaDatum(key="test", value=1.0)
    # @TODO: Fix when geojson is implemented
    schemas.MetaDatum(
        key="test", value=schemas.GeoJSON(type="test", coordinates=[])
    )

    with pytest.raises(TypeError) as e:
        schemas.MetaDatum(key=123, value=123)
    assert "should always be of type string" in str(e)

    # Test supported value types
    with pytest.raises(NotImplementedError):
        schemas.MetaDatum(key="test", value=(1, 2))
    assert "has unsupported type <class 'tuple'>"
    with pytest.raises(NotImplementedError):
        schemas.MetaDatum(key="test", value=[1, 2])
    assert "has unsupported type <class 'list'>"

    # Test special type with name=href
    schemas.MetaDatum(key="href", value="http://test")
    schemas.MetaDatum(key="href", value="https://test")
    with pytest.raises(ValueError) as e:
        schemas.MetaDatum(key="href", value="test")
    assert "`href` must start with http:// or https://" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.MetaDatum(key="href", value=1)
    assert "passed something other than 'str'" in str(e)

    # Check int to float conversion
    m = schemas.MetaDatum(key="test", value=1)
    assert isinstance(m.value, float)


def test_core_dataset():
    # valid
    schemas.Dataset(
        name="test",
        metadata=[schemas.MetaDatum(key="test", value=123)],
    )
    schemas.Dataset(
        name="test",
        id=None,
        metadata=[],
    )
    schemas.Dataset(
        name="test",
        id=1,
        metadata=[],
    )
    schemas.Dataset(
        name="test",
    )

    # test `__post_init__`
    with pytest.raises(AssertionError):
        schemas.Dataset(name=123)
    with pytest.raises(AssertionError):
        schemas.Dataset(name="123", id="123")
    with pytest.raises(AssertionError):
        schemas.Dataset(name="123", metadata=1)
    with pytest.raises(AssertionError):
        schemas.Dataset(name="123", metadata=[1])


def test_core_model():
    # valid
    schemas.Model(
        name="test",
        metadata=[schemas.MetaDatum(key="test", value=123)],
    )
    schemas.Model(
        name="test",
        id=None,
        metadata=[],
    )
    schemas.Model(
        name="test",
        id=1,
        metadata=[],
    )
    schemas.Model(
        name="test",
    )

    # test `__post_init__`
    with pytest.raises(AssertionError):
        schemas.Model(name=123)
    with pytest.raises(AssertionError):
        schemas.Model(name="123", id="123")
    with pytest.raises(AssertionError):
        schemas.Model(name="123", metadata=1)
    with pytest.raises(AssertionError):
        schemas.Model(name="123", metadata=[1])


def test_core_info():
    # @TODO: Not fully implemented
    schemas.Info()


def test_core_datum():
    schemas.Datum(uid="123")
    schemas.Datum(uid="123", metadata=[])
    schemas.Datum(uid="123", metadata=[schemas.MetaDatum(key="name", value=1)])

    # test `__post_init__`
    with pytest.raises(AssertionError):
        schemas.Datum(uid=123)
    with pytest.raises(AssertionError):
        schemas.Datum(uid="123", metadata=1)
    with pytest.raises(AssertionError):
        schemas.Datum(uid="123", metadata=[1])


def test_core_annotation(bbox, polygon, raster, metadatum):
    # valid
    schemas.Annotation(task_type=enums.TaskType.DETECTION, bounding_box=bbox)
    schemas.Annotation(task_type=enums.TaskType.DETECTION, polygon=polygon)
    schemas.Annotation(
        task_type=enums.TaskType.INSTANCE_SEGMENTATION, raster=raster
    )
    schemas.Annotation(
        task_type=enums.TaskType.SEMANTIC_SEGMENTATION, raster=raster
    )
    schemas.Annotation(
        task_type=enums.TaskType.DETECTION,
        bounding_box=bbox,
        polygon=polygon,
        raster=raster,
    )
    schemas.Annotation(task_type=enums.TaskType.CLASSIFICATION)
    schemas.Annotation(task_type=enums.TaskType.CLASSIFICATION, metadata=[])
    schemas.Annotation(
        task_type=enums.TaskType.CLASSIFICATION, metadata=[metadatum]
    )

    # test `__post_init__`
    with pytest.raises(ValueError) as e:
        schemas.Annotation(task_type="something")
    assert "is not a valid TaskType" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(
            task_type=enums.TaskType.DETECTION, bounding_box=polygon
        )
    assert "does not match" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(task_type=enums.TaskType.DETECTION, polygon=bbox)
    assert "does not match" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(
            task_type=enums.TaskType.DETECTION, multipolygon=bbox
        )
    assert "does not match" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(task_type=enums.TaskType.DETECTION, raster=bbox)
    assert "does not match" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(
            task_type=enums.TaskType.CLASSIFICATION, metadata=metadatum
        )
    assert "incorrect type" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(
            task_type=enums.TaskType.CLASSIFICATION, metadata=[123]
        )
    assert "incorrect type" in str(e)


def test_core_label():
    # valid
    l1 = schemas.Label(key="test", value="value")

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        schemas.Label(key=123, value="123")
    assert "key is not a `str`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Label(key="123", value=123)
    assert "value is not a `str`" in str(e)

    # test member fn `tuple`
    assert l1.tuple() == ("test", "value")

    # test member fn `__eq__`
    l2 = schemas.Label(key="test", value="value")
    assert l1 == l2

    # test member fn `__hash__`
    assert l1.__hash__() == l2.__hash__()


def test_core_scored_label():
    l1 = schemas.Label(key="test", value="value")
    l2 = schemas.Label(key="test", value="other")
    l3 = schemas.Label(key="other", value="value")

    # valid
    s1 = schemas.ScoredLabel(label=l1, score=0.5)
    s2 = schemas.ScoredLabel(label=l1, score=0.5)
    s3 = schemas.ScoredLabel(label=l1, score=0.1)
    s4 = schemas.ScoredLabel(label=l2, score=0.5)
    s5 = schemas.ScoredLabel(label=l3, score=0.5)

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        schemas.ScoredLabel(label="label", score=0.5)
    assert "label should be of type `velour.schemas.Label`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.ScoredLabel(label=l1, score="0.5")
    assert "score should be of type `float`" in str(e)

    # test property `key`
    assert l1.key == "test"

    # test property `value`
    assert l1.value == "value"

    # test member fn `__eq__`
    assert s1 == s2
    assert not s1 == s3
    assert not s1 == s4
    assert not s1 == s5
    assert not s1 == 123
    assert not s1 == "123"

    # test member fn `__eq__`
    assert not s1 != s2
    assert s1 != s3
    assert s1 != s4
    assert s1 != s5
    assert s1 != 123
    assert s1 != "123"

    # test member fn `__hash__`
    assert s1.__hash__() == s2.__hash__()
    assert s1.__hash__() != s3.__hash__()
    assert s1.__hash__() != s4.__hash__()
    assert s1.__hash__() != s5.__hash__()


def test_core_label_distribution():
    l1 = schemas.Label(key="test", value="value")

    # valid
    schemas.LabelDistribution(label=l1, count=10)


def test_core_scored_label_distribiution():
    l1 = schemas.Label(key="test", value="value")

    # valid
    schemas.ScoredLabelDistribution(label=l1, count=2, scores=[0.1, 0.5])


def test_core_annotation_distribution():
    # valid
    schemas.AnnotationDistribution(
        annotation_type=enums.AnnotationType.BOX,
        count=10,
    )


def test_core_groundtruth_annotation():
    l1 = schemas.Label(key="test", value="value")
    l2 = schemas.Label(key="test", value="other")
    l3 = schemas.Label(key="other", value="value")

    # valid
    schemas.Annotation(
        task_type=enums.TaskType.CLASSIFICATION,
        labels=[l1, l2, l3],
    )

    # test `__post_init__`
    with pytest.raises(ValueError) as e:
        schemas.Annotation(task_type="soemthing", labels=[l1])
    assert "is not a valid TaskType" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(task_type=enums.TaskType.CLASSIFICATION, labels=l1)
    assert "labels should be a list of `velour.schemas.Label`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Annotation(
            task_type=enums.TaskType.CLASSIFICATION, labels=[l1, l2, "label"]
        )
    assert (
        "elements of labels should be of type `velour.schemas.Label`" in str(e)
    )


def test_core_prediction_annotation():
    l1 = schemas.Label(key="test", value="value")
    l2 = schemas.Label(key="test", value="other")
    l3 = schemas.Label(key="other", value="value")
    s1 = schemas.ScoredLabel(label=l1, score=0.5)
    s2 = schemas.ScoredLabel(label=l2, score=0.5)
    s3 = schemas.ScoredLabel(label=l3, score=1.0)

    # valid
    schemas.ScoredAnnotation(
        task_type=enums.TaskType.CLASSIFICATION,
        scored_labels=[s1, s2, s3],
    )

    # test `__post_init__`
    with pytest.raises(ValueError) as e:
        schemas.ScoredAnnotation(
            task_type="something", scored_labels=[s1, s2, s3]
        )
    assert "is not a valid TaskType" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.ScoredAnnotation(
            task_type=enums.TaskType.CLASSIFICATION, scored_labels=s1
        )
    assert (
        "scored_labels should be a list of `velour.schemas.ScoredLabel`"
        in str(e)
    )
    with pytest.raises(TypeError) as e:
        schemas.ScoredAnnotation(
            task_type=enums.TaskType.CLASSIFICATION,
            scored_labels=[s1, s2, "label"],
        )
    assert (
        "elements of scored_labels should be of type `velour.schemas.ScoredLabel`"
        in str(e)
    )
    with pytest.raises(ValueError) as e:
        schemas.ScoredAnnotation(
            task_type=enums.TaskType.CLASSIFICATION, scored_labels=[s1, s3]
        )
    assert "for label key test got scores summing to 0.5" in str(e)


def test_core_groundtruth():
    label = schemas.Label(key="test", value="value")
    datum = schemas.Datum(uid="somefile")
    gts = [
        schemas.Annotation(
            task_type=enums.TaskType.CLASSIFICATION, labels=[label]
        ),
        schemas.Annotation(
            task_type=enums.TaskType.CLASSIFICATION, labels=[label]
        ),
    ]

    # valid
    schemas.GroundTruth(
        datum=datum,
        annotations=gts,
    )
    schemas.GroundTruth(datum=datum, annotations=gts, dataset_name="test")

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        schemas.GroundTruth(
            datum="datum",
            annotations=gts,
        )
    assert "datum should be type `velour.schemas.Datum`." in str(e)
    with pytest.raises(TypeError) as e:
        schemas.GroundTruth(
            datum=datum,
            annotations=gts[0],
        )
    assert (
        "annotations should be a list of `velour.schemas.Annotation`."
        in str(e)
    )
    with pytest.raises(TypeError) as e:
        schemas.GroundTruth(
            datum=datum,
            annotations=[gts[0], gts[1], "annotation"],
        )
    assert (
        "annotations list should contain only `velour.schemas.Annotation`."
        in str(e)
    )
    with pytest.raises(TypeError) as e:
        schemas.GroundTruth(
            datum=datum,
            annotations=gts,
            dataset_name=1234,
        )
    assert "dataset_name should be type `str`." in str(e)


def test_core_prediction():
    label = schemas.Label(key="test", value="value")
    scored_label = schemas.ScoredLabel(label=label, score=1.0)
    datum = schemas.Datum(uid="somefile")
    pds = [
        schemas.ScoredAnnotation(
            task_type=enums.TaskType.CLASSIFICATION,
            scored_labels=[scored_label],
        ),
        schemas.ScoredAnnotation(
            task_type=enums.TaskType.CLASSIFICATION,
            scored_labels=[scored_label],
        ),
    ]

    # valid
    schemas.Prediction(
        datum=datum,
        annotations=pds,
    )
    schemas.Prediction(datum=datum, annotations=pds, model_name="test")

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        schemas.Prediction(
            datum="datum",
            annotations=pds,
        )
    assert "datum should be type `velour.schemas.Datum`." in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Prediction(
            datum=datum,
            annotations=pds[0],
        )
    assert (
        "annotations should be a list of `velour.schemas.ScoredAnnotation`."
        in str(e)
    )
    with pytest.raises(TypeError) as e:
        schemas.Prediction(
            datum=datum,
            annotations=[pds[0], pds[1], "annotation"],
        )
    assert (
        "annotations list should contain only `velour.schemas.ScoredAnnotation`."
        in str(e)
    )
    with pytest.raises(TypeError) as e:
        schemas.Prediction(
            datum=datum,
            annotations=pds,
            model_name=1234,
        )
    assert "model_name should be type `str`." in str(e)
