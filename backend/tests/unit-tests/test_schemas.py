import os
import math
from base64 import b64encode
from tempfile import TemporaryDirectory

import numpy as np
import PIL.Image
import pytest
from pydantic import ValidationError

from velour_api import schemas, enums
from velour_api.schemas.core import _format_name, _format_uid


@pytest.fixture
def metadata() -> list[schemas.MetaDatum]:
    return [
        schemas.MetaDatum(
            name="m1",
            value="v1",
        ),
        schemas.MetaDatum(
            name="m2",
            value=0.1,
        ),
    ]


@pytest.fixture
def box_points() -> list[schemas.geometry.Point]:
    return [
        schemas.geometry.Point(x=-5, y=-5),
        schemas.geometry.Point(x=5, y=-5),
        schemas.geometry.Point(x=5, y=5),
        schemas.geometry.Point(x=-5, y=5),
    ]


@pytest.fixture
def rotated_box_points() -> list[schemas.geometry.Point]:
    """Same area and sides as box_points, but rotated 45 degrees."""
    d = 5.0 * math.sqrt(2)
    return [
        schemas.geometry.Point(x=0, y=d),
        schemas.geometry.Point(x=d, y=0),
        schemas.geometry.Point(x=0, y=-d),
        schemas.geometry.Point(x=-d, y=0),
    ]


@pytest.fixture
def skewed_box_points() -> list[schemas.geometry.Point]:
    """Skewed box_points."""
    return [
        schemas.geometry.Point(x=0, y=0),
        schemas.geometry.Point(x=10, y=0),
        schemas.geometry.Point(x=15, y=10),
        schemas.geometry.Point(x=5, y=10),
    ]


@pytest.fixture
def component_polygon_box(box_points) -> schemas.geometry.BasicPolygon:
    return schemas.geometry.BasicPolygon(
        points=box_points,
    )
    

@pytest.fixture
def component_polygon_rotated_box(rotated_box_points) -> schemas.geometry.BasicPolygon:
    return schemas.geometry.BasicPolygon(
        points=rotated_box_points,
    )


@pytest.fixture
def component_polygon_skewed_box(skewed_box_points) -> schemas.geometry.BasicPolygon:
    return schemas.geometry.BasicPolygon(
        points=skewed_box_points,
    )
    

@pytest.fixture
def bbox(component_polygon_box) -> schemas.BoundingBox:
    return schemas.BoundingBox(
        polygon=component_polygon_box,
    )


@pytest.fixture
def polygon(component_polygon_box) -> schemas.Polygon:
    return schemas.Polygon(boundary=component_polygon_box)


def _create_b64_mask(mode: str, ext: str = ".png", size=(20,20)) -> str:
    with TemporaryDirectory() as tempdir:
        img = PIL.Image.new(mode=mode, size=size)
        img_path = os.path.join(tempdir, f"img.{ext}")
        img.save(img_path)

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        return b64encode(img_bytes).decode()

@pytest.fixture
def raster() -> schemas.Raster:
    """
    Creates a 2d numpy of bools of shape:
    | T  F |
    | F  T |
    """
    mask = "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUAQAAAACl8iCgAAAAF0lEQVR4nGP4f4CBiYGBIGZgsP9AjDoAuysDE0GVDN8AAAAASUVORK5CYII="
    height = 20
    width = 20
    return schemas.Raster(mask=mask, height=height, width=width)


@pytest.fixture
def labels() -> list[schemas.Label]:
    return [
        schemas.Label(key="k1", value="v1"),
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k2", value="v3"),
        schemas.Label(key="k3", value="v4"),
    ]


@pytest.fixture
def scored_labels(labels) -> list[schemas.ScoredLabel]:
    return [
        schemas.ScoredLabel(label=labels[0], score=0.1),
        schemas.ScoredLabel(label=labels[1], score=0.9),
        schemas.ScoredLabel(label=labels[2], score=1.0),
        schemas.ScoredLabel(label=labels[3], score=1.0),
    ]


@pytest.fixture
def groundtruth_annotations(labels) -> list[schemas.GroundTruthAnnotation]:
    return [
        schemas.GroundTruthAnnotation(
            labels=[labels[0]],
            annotation=schemas.Annotation(task_type=enums.TaskType.CLASSIFICATION)
        ),
        schemas.GroundTruthAnnotation(
            labels=[labels[2]],
            annotation=schemas.Annotation(task_type=enums.TaskType.CLASSIFICATION)
        ),
        schemas.GroundTruthAnnotation(
            labels=[labels[3]],
            annotation=schemas.Annotation(task_type=enums.TaskType.CLASSIFICATION)
        ),
    ]


@pytest.fixture
def predicted_annotations(scored_labels) -> list[schemas.PredictedAnnotation]:
    return [
        schemas.PredictedAnnotation(
            scored_labels=[scored_labels[0], scored_labels[1]],
            annotation=schemas.Annotation(task_type=enums.TaskType.CLASSIFICATION),
        ),
        schemas.PredictedAnnotation(
            scored_labels=[scored_labels[2]],
            annotation=schemas.Annotation(task_type=enums.TaskType.CLASSIFICATION),
        ),
        schemas.PredictedAnnotation(
            scored_labels=[scored_labels[3]],
            annotation=schemas.Annotation(task_type=enums.TaskType.CLASSIFICATION),
        ),
    ]


""" velour_api.schemas.core """


def test_core__format_name():
    assert _format_name("dataset1") == "dataset1"
    assert _format_name("dataset-1") == "dataset-1"
    assert _format_name("dataset_1") == "dataset_1"
    assert _format_name("data!@#$%^&*()'set_1") == "dataset_1"


def test_core__format_uid():
    assert _format_uid("uid1") == "uid1"
    assert _format_uid("uid-1") == "uid-1"
    assert _format_uid("uid_1") == "uid_1"
    assert _format_uid("uid1.png") == "uid1.png"
    assert _format_uid("folder/uid1.png") == "folder/uid1.png"
    assert _format_uid("uid!@#$%^&*()'_1") == "uid_1"


def test_metadata_Metadatum():
    # valid
    schemas.MetaDatum(
        name="name",
        value="value"
    )
    schemas.MetaDatum(
        name="name",
        value=123,
    )
    schemas.MetaDatum(
        name="name",
        value=123.0,
    )
    # @TODO: After implement geojson
    # schemas.MetaDatum(
    #     name="name",
    #     value=schemas.GeoJSON(),
    # )

    # test property `name`
    with pytest.raises(ValidationError):
        schemas.MetaDatum(
            name=("name",),
            value=123,
        )

    # test property `value`
    with pytest.raises(ValidationError):
        schemas.MetaDatum(
            name="name",
            value=[1,2,3],
        )
    with pytest.raises(ValidationError):
        schemas.MetaDatum(
            name="name",
            value=(1,2,3),
        )
    with pytest.raises(ValidationError):
        schemas.MetaDatum(
            name="name",
            value=schemas.geometry.Point(x=1,y=1),
        )


def test_core_Dataset(metadata):
    # valid
    schemas.Dataset(
        name="dataset1"
    )
    schemas.Dataset(
        name="dataset1",
        metadata=[],
    )
    schemas.Dataset(
        name="dataset1",
        metadata=metadata,
    )
    schemas.Dataset(
        id=1,
        name="dataset1",
        metadata=metadata,
    )

    # test property `name`
    with pytest.raises(ValidationError):
        schemas.Dataset(
            name=(12,),
        )
    with pytest.raises(ValidationError):
        schemas.Dataset(
            name="dataset@"
        )

    # test property `metadata`
    with pytest.raises(ValidationError):
        schemas.Dataset(
            name="123",
            metadata=metadata[0],
        )
    with pytest.raises(ValidationError):
        schemas.Dataset(
            name="123",
            metadata=[metadata[0], "123"],
        )

    # test property `id`
    with pytest.raises(ValidationError):
        schemas.Dataset(
            id="value",
            name="123",
            metadata=[metadata[0], "123"],
        )


def test_core_Model(metadata):
    # valid
    schemas.Model(
        name="model1"
    )
    schemas.Model(
        name="model1",
        metadata=[],
    )
    schemas.Model(
        name="model1",
        metadata=metadata,
    )
    schemas.Model(
        id=1,
        name="model1",
        metadata=metadata,
    )

    # test property `name`
    with pytest.raises(ValidationError):
        schemas.Model(
            name=(12,),
        )
    with pytest.raises(ValidationError):
        schemas.Dataset(
            name="model@"
        )

    # test property `metadata`
    with pytest.raises(ValidationError):
        schemas.Model(
            name="123",
            metadata=metadata[0],
        )
    with pytest.raises(ValidationError):
        schemas.Model(
            name="123",
            metadata=[metadata[0], "123"],
        )

    # test property `id`
    with pytest.raises(ValidationError):
        schemas.Model(
            id="value",
            name="123",
            metadata=[metadata[0], "123"],
        )


def test_core_Datum(metadata):
    # valid
    schemas.Datum(
        uid="123",
    )
    schemas.Datum(
        uid=123,
    )

    # test property `uid`
    with pytest.raises(ValidationError):
        schemas.Datum(
            uid=("uid",)
        )
    with pytest.raises(ValidationError):
        schemas.Datum(
            uid="uid@"
        )

    # test property `metadata`
    with pytest.raises(ValidationError):
        schemas.Datum(
            uid="123",
            metadata=metadata[0],
        )
    with pytest.raises(ValidationError):
        schemas.Datum(
            uid="123",
            metadata=[metadata[0], "123"],
        )


def test_core_Annotation(metadata, bbox, polygon, raster):
    # valid 
    schemas.Annotation(
        task_type=enums.TaskType.CLASSIFICATION,
    )
    schemas.Annotation(
        task_type=enums.TaskType.DETECTION,
        metadata=[],
    )
    schemas.Annotation(
        task_type=enums.TaskType.INSTANCE_SEGMENTATION,
        metadata=metadata,
    )
    schemas.Annotation(
        task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
        metadata=[],
        bounding_box=bbox,
        polygon=polygon,
        raster=raster,
    )
    schemas.Annotation(
        task_type="classification",
    )
    schemas.Annotation(
        task_type="detection",
    )
    schemas.Annotation(
        task_type="instance_segmentation",
    )
    schemas.Annotation(
        task_type="semantic_segmentation",
    )

    # test property `task_type`
    with pytest.raises(ValidationError):
        schemas.Annotation(
            task_type="custom",
        )
    
    # test property `metadata`
    with pytest.raises(ValidationError):
        schemas.Annotation(
            task_type="classification",
            metadata=metadata[0],
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            task_type="classification",
            metadata=[metadata[0], "123"],
        )

    # test geometric properties 
    with pytest.raises(ValidationError):
        schemas.Annotation(
            task_type=enums.TaskType.DETECTION,
            bounding_box=polygon,
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            task_type=enums.TaskType.DETECTION,
            polygon=bbox,
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            task_type=enums.TaskType.DETECTION,
            multipolygon=bbox,
        )
    with pytest.raises(ValidationError) as e:
        schemas.Annotation(
            task_type=enums.TaskType.DETECTION,
            raster=bbox,
        )


def test_core_GroundtruthAnnotation(labels):
    # valid
    gt = schemas.GroundTruthAnnotation(
        labels=labels,
        annotation=schemas.Annotation(
            task_type=enums.TaskType.CLASSIFICATION,
        )
    )

    # test property `labels`
    with pytest.raises(ValidationError):
        schemas.GroundTruthAnnotation(
            labels=labels[0],
            annotation=schemas.Annotation(
                task_type=enums.TaskType.CLASSIFICATION,
            )
        )
    with pytest.raises(ValidationError):
        schemas.GroundTruthAnnotation(
            labels=[labels[0], 123],
            annotation=schemas.Annotation(
                task_type=enums.TaskType.CLASSIFICATION,
            )
        )   
    with pytest.raises(ValidationError):
        schemas.GroundTruthAnnotation(
            labels=[],
            annotation=schemas.Annotation(
                task_type=enums.TaskType.CLASSIFICATION,
            )
        )   
    assert gt.labels == labels

    # test property `annotation`
    with pytest.raises(ValidationError):
        schemas.GroundTruthAnnotation(
            labels=labels,
            annotation="classification"
        )
    assert gt.annotation == schemas.Annotation(
        task_type=enums.TaskType.CLASSIFICATION,
    )
    

def test_core_PredictedAnnotation(scored_labels):
    # valid
    pd = schemas.PredictedAnnotation(
        scored_labels=scored_labels,
        annotation=schemas.Annotation(task_type=enums.TaskType.CLASSIFICATION),
    )

    # test property `scored_labels`
    with pytest.raises(ValidationError) as e:
        schemas.PredictedAnnotation(
            scored_labels=scored_labels[0],
            annotation=schemas.Annotation(task_type=enums.TaskType.CLASSIFICATION),
        )
    assert "value is not a valid list" in str(e.value.errors()[0]["msg"])
    with pytest.raises(ValidationError) as e:
        schemas.PredictedAnnotation(
            scored_labels=scored_labels[1:],
            annotation=schemas.Annotation(task_type=enums.TaskType.CLASSIFICATION),
        )
    assert "prediction scores must sum to 1" in str(e.value.errors()[0]["msg"])
    assert pd.scored_labels == scored_labels

    # test property `annotation`
    with pytest.raises(ValidationError):
        schemas.PredictedAnnotation(
            scored_labels=scored_labels,
            annotation="annotation"
        )
    assert pd.annotation == schemas.Annotation(task_type=enums.TaskType.CLASSIFICATION)


def test_core_Groundtruth(metadata, groundtruth_annotations):
    # valid
    gt = schemas.GroundTruth(
        dataset_name="name1",
        datum=schemas.Datum(uid="uid"),
        annotations=groundtruth_annotations,
    )

    # test property `dataset_name`
    assert gt.dataset_name == "name1"
    with pytest.raises(ValidationError):
        schemas.GroundTruth(
            dataset_name=("name",),
            datum=schemas.Datum(uid="uid"),
            annotations=groundtruth_annotations,
        )
    with pytest.raises(ValidationError):
        schemas.GroundTruth(
            dataset_name="name@#$#@",
            datum=schemas.Datum(uid="uid"),
            annotations=groundtruth_annotations,
        )

    # test property `datum`
    assert gt.datum == schemas.Datum(uid="uid")
    with pytest.raises(ValidationError):
        schemas.GroundTruth(
            dataset_name="name",
            datum="datum_uid",
            annotations=groundtruth_annotations,
        )

    # test property `annotations`
    assert gt.annotations == groundtruth_annotations
    with pytest.raises(ValidationError):
        schemas.GroundTruth(
            dataset_name="name",
            datum=schemas.Datum(uid="uid"),
            annotations="annotation",
        )
    with pytest.raises(ValidationError):
        schemas.GroundTruth(
            dataset_name="name",
            datum=schemas.Datum(uid="uid"),
            annotations=[],
        )
    with pytest.raises(ValidationError):
        schemas.GroundTruth(
            dataset_name="name",
            datum=schemas.Datum(uid="uid"),
            annotations=[groundtruth_annotations[0], 1234],
        )


def test_core_Prediction(metadata, predicted_annotations):
    # valid
    md = schemas.Prediction(
        model_name="name1",
        datum=schemas.Datum(uid="uid"),
        annotations=predicted_annotations,
    )

    # test property `model_name`
    assert md.model_name == "name1"
    with pytest.raises(ValidationError):
        schemas.Prediction(
            model_name=("name",),
            datum=schemas.Datum(uid="uid"),
            annotations=predicted_annotations,
        )
    with pytest.raises(ValidationError):
        schemas.Prediction(
            model_name="name@#$#@",
            datum=schemas.Datum(uid="uid"),
            annotations=predicted_annotations,
        )

    # test property `datum`
    assert md.datum == schemas.Datum(uid="uid")
    with pytest.raises(ValidationError):
        schemas.Prediction(
            model_name="name",
            datum="datum_uid",
            annotations=predicted_annotations,
        )

    # test property `annotations`
    assert md.annotations == predicted_annotations
    with pytest.raises(ValidationError):
        schemas.Prediction(
            model_name="name",
            datum=schemas.Datum(uid="uid"),
            annotations="annotation",
        )
    with pytest.raises(ValidationError):
        schemas.Prediction(
            model_name="name",
            datum=schemas.Datum(uid="uid"),
            annotations=[],
        )
    with pytest.raises(ValidationError):
        schemas.Prediction(
            model_name="name",
            datum=schemas.Datum(uid="uid"),
            annotations=[predicted_annotations[0], 1234],
        )


""" velour_api.schemas.metadata """


# @TODO
def test_metadata_Image():
    pass


""" velour_api.schemas.geometry """


def test_geometry_Point():
    # valid
    p1 = schemas.geometry.Point(x=3.14, y=-3.14)
    p2 = schemas.geometry.Point(x=3.14, y=-3.14)
    p3 = schemas.geometry.Point(x=-3.14, y=3.14)

    # test properties
    with pytest.raises(ValidationError):
        schemas.geometry.Point(x="test",y=0)
    with pytest.raises(ValidationError):
        schemas.geometry.Point(x=0,y="test")

    # test member fn `__str__`
    assert str(p1) == "(3.14, -3.14)"

    # test member fn `__hash__`
    assert p1.__hash__() == p2.__hash__()
    assert p1.__hash__() != p3.__hash__()

    # test member fn `__eq__`
    assert p1 == p2
    assert not p1 == p3

    # test member fn `__neq__`
    assert not p1 != p2
    assert p1 != p3

    # test member fn `__neg__`
    assert p1 == -p3

    # test member fn `__add__`
    assert p1 + p3 == schemas.geometry.Point(x=0, y=0)

    # test member fn `__sub__`
    assert p1 - p2 == schemas.geometry.Point(x=0, y=0)

    # test member fn `__iadd__`
    p4 = p3
    p4 += p1
    assert p4 == schemas.geometry.Point(x=0, y=0)
    
    # test member fn `__isub__`
    p4 = p3
    p4 -= p3
    assert p4 == schemas.geometry.Point(x=0, y=0)

    # test member fn `dot`
    assert p1.dot(p2) == (3.14**2) * 2


def test_geometry_LineSegment(box_points):
    # valid 
    l1 = schemas.geometry.LineSegment(
        points=(
            box_points[0],
            box_points[1],
        )
    )
    l2 = schemas.geometry.LineSegment(
        points=(
            box_points[-1],
            box_points[0],
        )
    )
    l3 = schemas.geometry.LineSegment(
        points=(
            box_points[2],
            box_points[3],
        )
    )

    # test property `points`
    with pytest.raises(ValidationError):
        schemas.geometry.LineSegment(points="points")
    with pytest.raises(ValidationError):
        schemas.geometry.LineSegment(points=box_points[0])
    with pytest.raises(ValidationError):
        schemas.geometry.LineSegment(points=(box_points[0],))
    with pytest.raises(ValidationError):
        schemas.geometry.LineSegment(points=(box_points[0],box_points[1], box_points[2]))
    with pytest.raises(ValidationError):
        schemas.geometry.LineSegment(points=(1,2))

    # test member fn `delta_xy`
    assert l1.delta_xy() == box_points[0] - box_points[1]

    # test member fn `parallel`
    assert not l1.parallel(l2)
    assert l1.parallel(l3)
    assert l3.parallel(l1)

    # test member fn `perpendicular`
    assert l2.perpendicular(l3)
    assert l3.perpendicular(l2)
    assert not l1.perpendicular(l3)


# @TODO
def test_geometry_BasicPolygon(box_points):
    # valid
    poly = schemas.geometry.BasicPolygon(
        points=box_points,
    )

    # test property `points`
    with pytest.raises(ValidationError):
        schemas.geometry.BasicPolygon(points=box_points[0])
    with pytest.raises(ValidationError):
        schemas.geometry.BasicPolygon(points=[])
    with pytest.raises(ValidationError):
        schemas.geometry.BasicPolygon(points=[
            box_points[0],
            box_points[1],
        ])
    with pytest.raises(ValidationError):
        schemas.geometry.BasicPolygon(points=[
            box_points[0],
            box_points[1],
            (1,3),
        ])

    # test member fn `left` @TODO
    
    # test member fn `right` @TODO
    
    # test member fn `top` @TODO
    
    # test member fn `bottom` @TODO
    
    # test member fn `width` @TODO
    
    # test member fn `height` @TODO

    # test member fn `segments`
    plist = box_points + [box_points[0]]
    assert poly.segments == [
        schemas.geometry.LineSegment(points=(plist[i], plist[i+1]))
        for i in range(len(plist)-1)
    ]

    # test member fn `__str__`
    assert str(poly) == "((-5.0,-5.0),(5.0,-5.0),(5.0,5.0),(-5.0,5.0),(-5.0,-5.0))"

    # test member fn `wkt`
    assert poly.wkt() == "POLYGON ((-5.0 -5.0, 5.0 -5.0, 5.0 5.0, -5.0 5.0, -5.0 -5.0))"


def test_geometry_Polygon(
    component_polygon_box,
    component_polygon_rotated_box, 
    component_polygon_skewed_box,
):
    # valid
    p1 = schemas.Polygon(
        boundary=component_polygon_box,
    )
    schemas.Polygon(
        boundary=component_polygon_skewed_box,
        holes=[],
    )
    p2 = schemas.Polygon(
        boundary=component_polygon_skewed_box,
        holes=[component_polygon_box],
    )
    schemas.Polygon(
        boundary=component_polygon_skewed_box,
        holes=[component_polygon_box, component_polygon_rotated_box],
    )

    # test property `boundary`
    with pytest.raises(ValidationError):            # type checking
        schemas.Polygon(
            boundary="component_polygon_skewed_box",
        )
    with pytest.raises(ValidationError):
        schemas.Polygon(
            boundary=[component_polygon_box],
        )

    # test property `holes`
    with pytest.raises(ValidationError):            # type checking
        schemas.Polygon(
            boundary=component_polygon_box,
            holes=component_polygon_skewed_box,
        )
    with pytest.raises(ValidationError):
        schemas.Polygon(
            boundary=component_polygon_box,
            holes=[component_polygon_skewed_box, 123],
        )

    # test member fn `__str__`
    assert str(p1) == "(((-5.0,-5.0),(5.0,-5.0),(5.0,5.0),(-5.0,5.0),(-5.0,-5.0)))"
    assert str(p2) == "(((0.0,0.0),(10.0,0.0),(15.0,10.0),(5.0,10.0),(0.0,0.0)),((-5.0,-5.0),(5.0,-5.0),(5.0,5.0),(-5.0,5.0),(-5.0,-5.0)))"
    
    # test member fn `wkt`
    assert p1.wkt() == "POLYGON ((-5.0 -5.0, 5.0 -5.0, 5.0 5.0, -5.0 5.0, -5.0 -5.0))"
    assert p2.wkt() == "POLYGON ((0.0 0.0, 10.0 0.0, 15.0 10.0, 5.0 10.0, 0.0 0.0), (-5.0 -5.0, 5.0 -5.0, 5.0 5.0, -5.0 5.0, -5.0 -5.0))"


def test_geometry_MultiPolygon(
    component_polygon_box,
    component_polygon_rotated_box, 
    component_polygon_skewed_box
):
    p1 = schemas.Polygon(boundary=component_polygon_rotated_box)
    p2 = schemas.Polygon(boundary=component_polygon_skewed_box, holes=[component_polygon_box])

    # valid
    mp1 = schemas.MultiPolygon(polygons=[p1])
    mp2 = schemas.MultiPolygon(polygons=[p1, p2])

    # test property `polygons`
    with pytest.raises(ValidationError):            # type checking
        schemas.MultiPolygon(polygons=component_polygon_box)
    with pytest.raises(ValidationError):
        schemas.MultiPolygon(polygons=p1)
    with pytest.raises(ValidationError):
        schemas.MultiPolygon(polygons=[component_polygon_box])
    with pytest.raises(ValidationError):
        schemas.MultiPolygon(polygons=[p1, component_polygon_box])

    # test member fn `wkt`
    assert mp1.wkt() == "MULTIPOLYGON (((0.0 7.0710678118654755, 7.0710678118654755 0.0, 0.0 -7.0710678118654755, -7.0710678118654755 0.0, 0.0 7.0710678118654755)))"
    assert mp2.wkt() == "MULTIPOLYGON (((0.0 7.0710678118654755, 7.0710678118654755 0.0, 0.0 -7.0710678118654755, -7.0710678118654755 0.0, 0.0 7.0710678118654755)), ((0.0 0.0, 10.0 0.0, 15.0 10.0, 5.0 10.0, 0.0 0.0), (-5.0 -5.0, 5.0 -5.0, 5.0 5.0, -5.0 5.0, -5.0 -5.0)))"


# @TODO
def test_geometry_BoundingBox(
    component_polygon_box,
    component_polygon_rotated_box,
    component_polygon_skewed_box,
):
    # valid
    bbox1 = schemas.BoundingBox(
        polygon=component_polygon_box,
    )
    bbox2 = schemas.BoundingBox(
        polygon=component_polygon_rotated_box
    )
    bbox3 = schemas.BoundingBox(
        polygon=component_polygon_skewed_box
    )

    # test property `polygon`
    with pytest.raises(ValidationError):            # type checking
        schemas.BoundingBox(polygon=1234)
    with pytest.raises(ValidationError):
        schemas.BoundingBox(polygon=component_polygon_box.points[0])
    with pytest.raises(ValidationError):
        schemas.BoundingBox(polygon=[component_polygon_box])
    with pytest.raises(ValidationError):
        box_plus_one = schemas.geometry.BasicPolygon(
            points=component_polygon_box.points + [schemas.geometry.Point(x=15,y=15)]
        )
        schemas.BoundingBox(polygon=box_plus_one)   # check for 4 unique points


    # test classmethod `from_extrema`
    assert schemas.BoundingBox.from_extrema(
        xmin=component_polygon_box.left,
        ymin=component_polygon_box.bottom,
        xmax=component_polygon_box.right,
        ymax=component_polygon_box.top,
    ).polygon == component_polygon_box

    # test member fn `left` @TODO
    
    # test member fn `right` @TODO

    # test member fn `top` @TODO
    
    # test member fn `bottom` @TODO

    # test member fn `width` @TODO

    # test member fn `height` @TODO

    # test member fn `is_rectangular`
    assert bbox1.is_rectangular()
    assert bbox2.is_rectangular()
    assert not bbox3.is_rectangular()

    # test member fn `is_rotated`
    assert not bbox1.is_rotated()
    assert bbox2.is_rotated()
    assert not bbox3.is_rotated()

    # test member fn `is_skewed`
    assert not bbox1.is_skewed()
    assert not bbox2.is_skewed()
    assert bbox3.is_skewed()

    # test member fn `wkt`
    assert bbox1.wkt() == "POLYGON ((-5.0 -5.0, 5.0 -5.0, 5.0 5.0, -5.0 5.0, -5.0 -5.0))"
    assert bbox2.wkt() == "POLYGON ((0.0 7.0710678118654755, 7.0710678118654755 0.0, 0.0 -7.0710678118654755, -7.0710678118654755 0.0, 0.0 7.0710678118654755))"
    assert bbox3.wkt() == "POLYGON ((0.0 0.0, 10.0 0.0, 15.0 10.0, 5.0 10.0, 0.0 0.0))"


def test_geometry_Raster(raster):
    # valid
    height = 20
    width = 20
    mask = _create_b64_mask(mode="1", size=(height, width))
    assert schemas.Raster(
        mask=mask,
        height=height,
        width=width,
    )

    # test property `mask`
    with pytest.raises(PIL.UnidentifiedImageError):
        # not any string can be passed
        schemas.Raster(mask="text", height=height, width=width)
    with pytest.raises(ValueError) as exc_info:
        base64_mask = _create_b64_mask(mode="RGB", ext="png", size=(width, height))
        schemas.Raster(
            mask=base64_mask,   # only supports binary images
            height=height, width=width,
        )
    assert "Expected image mode to be binary but got mode" in str(exc_info)
    with pytest.raises(ValueError) as exc_info:
        base64_mask = _create_b64_mask(mode="1", ext="jpg", size=(width, height))
        schemas.Raster(
            mask=base64_mask,   # Check we get an error if the format is not PNG
            height=height, width=width,
        )
    assert "Expected image format PNG but got" in str(exc_info)

    # test property `shape`
    with pytest.raises(ValidationError):
        mask = _create_b64_mask(mode="1", ext="PNG", size=(width, height))
        schemas.Raster(
            mask=mask,
            shape="shape",      # Incorrect type
        )
    with pytest.raises(ValidationError):
        mask = _create_b64_mask(mode="1", ext="PNG", size=(width, height))
        schemas.Raster(
            mask=mask,
            shape=(20,20,20),   # Incorrectly sized tuple
        )
    with pytest.raises(ValidationError) as e:
        mask = _create_b64_mask(mode="1", ext="PNG", size=(width, height))
        schemas.Raster(
            mask=mask,
            shape=(21,21),      # size mismatch
        )


""" velour_api.schemas.geojson """


# @TODO
def test_geojson_GeoJSON():
    pass


# @TODO
def test_geojson_GeoJSONPoint():
    pass


# @TODO
def test_geojson_GeoJSONPolygon():
    pass


# @TODO
def test_geojson_GeoJSONMultiPolygon():
    pass


""" velour_api.schemas.info """


# @TODO
def test_info_LabelDistribution():
    pass


# @TODO
def test_info_ScoredLabelDistribution():
    pass


# @TODO
def test_info_AnnotationDistribution():
    pass


""" velour_api.schemas.jobs """


# @TODO
def test_info_Job():
    pass


""" velour_api.schemas.label """


def test_label_Label():
    # valid
    l1 = schemas.Label(key="k1", value="v1")
    l2 = schemas.Label(key="k2", value="v2")

    # test property `key`
    with pytest.raises(ValidationError):
        schemas.Label(key=("k1",), value="v1")

    # test property `value`
    with pytest.raises(ValidationError):
        schemas.Label(key="k1", value=("v1",))

    # test classmethod fn `from_key_value_tuple`
    assert schemas.Label.from_key_value_tuple(("k1", "v1")) == l1
    assert schemas.Label.from_key_value_tuple(("1", "1")) != l1

    # test member fn `__eq__`
    assert l1 == l1
    assert not l1 == l2

    # test member fn `__hash__`
    assert l1.__hash__() == l1.__hash__()
    assert l1.__hash__() != l2.__hash__()


def test_label_ScoredLabel():
    l = schemas.Label(key="k1", value="v1")

    # valid
    schemas.ScoredLabel(label=l, score=0.5)
    
    # test property `label`
    with pytest.raises(ValidationError):
        schemas.ScoredLabel(label="label", score=0.5)

    # test property `score`
    with pytest.raises(ValidationError):
        schemas.ScoredLabel(label=l, score="score")

    
""" velour_api.schemas.auth """ 


# @TODO
def test_auth_User():
    # valid
    schemas.User(email="somestring")


""" velour_api.schemas.metrics """


# @TODO
def test_metrics_EvaluationSettings():
    pass

# @TODO
def test_metrics_APRequest():
    pass

# @TODO
def test_metrics_CreateAPMetricsResponse():
    pass

# @TODO
def test_metrics_CreateClfMetricsResponse():
    pass

# @TODO
def test_metrics_Job():
    pass

# @TODO
def test_metrics_ClfMetricsRequest():
    pass

# @TODO
def test_metrics_Metric():
    pass

# @TODO
def test_metrics_APMetric():
    pass

# @TODO
def test_metrics_APMetricAveragedOverIOUs():
    pass

# @TODO
def test_metrics_mAPMetric():
    pass

# @TODO
def test_metrics_mAPMetricAveragedOverIOUs():
    pass

# @TODO
def test_metrics_ConfusionMatrixEntry():
    pass

# @TODO
def test_metrics__BaseConfusionMatrix():
    pass

# @TODO
def test_metrics_ConfusionMatrix():
    pass

# @TODO
def test_metrics_ConfusionMatrixResponse():
    pass

# @TODO
def test_metrics_AccuracyMetric():
    pass

# @TODO
def test_metrics__PrecisionRecallF1Base():
    pass

# @TODO
def test_metrics_PrecisionMetric():
    pass

# @TODO
def test_metrics_RecallMetric():
    pass

# @TODO
def test_metrics_F1Metric():
    pass

# @TODO
def test_metrics_ROCAUCMetric():
    pass
