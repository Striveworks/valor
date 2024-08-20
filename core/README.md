# valor_core: Compute classification, object detection, and segmentation metrics locally.

Valor is a centralized evaluation store which makes it easy to measure, explore, and rank model performance. Valor empowers data scientists and engineers to evaluate the performance of their machine learning pipelines and use those evaluations to make better modeling decisions in the future.

`valor_core` is the start of a new backbone for Valor's metric calculations. In the future, the Valor API will import `valor_core`'s evaluation functions in order to efficiently compute its classification, object detection, and segmentation metrics. This module offers a few advantages over the existing `valor` evaluation implementations, including:
- The ability to calculate metrics locally, without running separate database and API services
- Faster compute times due to the use of vectors and arrays
- Easier testing, debugging, and benchmarking due to the separation of concerns between evaluation computations and Postgres operations (e.g., filtering, querying)

Valor is maintained by Striveworks, a cutting-edge MLOps company based out of Austin, Texas. We'd love to learn more about your interest in Valor and answer any questions you may have; please don't hesitate to reach out to us on [Slack](https://striveworks-public.slack.com/join/shared_invite/zt-1a0jx768y-2J1fffN~b4fXYM8GecvOhA#/shared-invite/email) or [GitHub](https://github.com/striveworks/valor).

For more information, please see our [user docs](https://striveworks.github.io/valor/).

## Usage

### Passing Lists of GroundTruth and Prediction Objects

The first way to use `valor_core` is to pass a list of groundtruth and prediction objects to an `evaluate_...` function, like so:

```python

groundtruths = [
    schemas.GroundTruth(
            datum=img1,
            annotations=...
     ), …
]
predictions = [
    schemas.Prediction(
            datum=img1,
            annotations=...
     ), …
]

evaluation = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
        metrics_to_return=[
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
        pr_curve_iou_threshold=0.5,
        pr_curve_max_examples=1,
    )
```

### Passing DataFrames

The second way to use `valor_core` is to pass in a dataframe of groundtruths and predictions:

```python

groundtruth_df = pd.DataFrame(
        [
            {
                "datum_id": 1,
                "datum_uid": "uid1",
                "id": 1,
                "annotation_id": 1,
                "label_id": 1,
                "label_key": "k1",
                "label_value": "v1",
                "is_instance": True,
                "grouper_key": "k1",
                "polygon": schemas.Polygon.from_dict(
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [[10, 10], [60, 10], [60, 40], [10, 40], [10, 10]]
                        ],
                    }
                ),
                "raster": None,
                "bounding_box": None,
            },
            {
                "datum_id": 1,
                "datum_uid": "uid1",
                "id": 2,
                "annotation_id": 2,
                "label_id": 2,
                "label_key": "k2",
                "label_value": "v2",
                "is_instance": True,
                "grouper_key": "k2",
                "polygon": schemas.Polygon.from_dict(
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [87, 10],
                                [158, 10],
                                [158, 820],
                                [87, 820],
                                [87, 10],
                            ]
                        ],
                    }
                ),
                "raster": None,
                "bounding_box": None,
            },
            {
                "datum_id": 2,
                "datum_uid": "uid2",
                "id": 3,
                "annotation_id": 3,
                "label_id": 1,
                "label_key": "k1",
                "label_value": "v1",
                "is_instance": True,
                "grouper_key": "k1",
                "polygon": schemas.Polygon.from_dict(
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [[15, 0], [70, 0], [70, 20], [15, 20], [15, 0]]
                        ],
                    }
                ),
                "raster": None,
                "bounding_box": None,
            },
        ]
)
prediction_df = pd.DataFrame(
    [
        {
            "id": 1,
            "annotation_id": 4,
            "score": 0.3,
            "datum_id": 1,
            "datum_uid": "uid1",
            "label_id": 1,
            "label_key": "k1",
            "label_value": "v1",
            "is_instance": True,
            "grouper_key": "k1",
            "polygon": schemas.Polygon.from_dict(
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[10, 10], [60, 10], [60, 40], [10, 40], [10, 10]]
                    ],
                }
            ),
            "raster": None,
            "bounding_box": None,
        },
        {
            "id": 2,
            "annotation_id": 5,
            "score": 0.98,
            "datum_id": 2,
            "datum_uid": "uid2",
            "label_id": 2,
            "label_key": "k2",
            "label_value": "v2",
            "is_instance": True,
            "grouper_key": "k2",
            "polygon": schemas.Polygon.from_dict(
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[15, 0], [70, 0], [70, 20], [15, 20], [15, 0]]
                    ],
                }
            ),
            "raster": None,
            "bounding_box": None,
        },
    ]
)

evaluation = evaluate_detection(
        groundtruths=groundtruth_df,
        predictions=prediction_df,
        metrics_to_return=[
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
        pr_curve_iou_threshold=0.5,
        pr_curve_max_examples=1,
    )
```

## Using a Data Manager

Finally, you can use a manager class (i.e., `ValorDetectionManager`) to run your evaluation. The advantage to using a manager class is a) you won't have to keep all annotation types in memory in a large list and b) we can pre-compute certain columns (i.e., `iou`) in advance of the `.evaluate()` call.


```python
manager = valor_core.ValorDetectionManager(...)
img1 = schemas.Datum(
        uid="uid1",
        metadata={
            "height": image_height,
            "width": image_width,
        },
    )
groundtruths = [
    schemas.GroundTruth(
            datum=img1,
            annotations=...
     ), …
]
predictions = [
    schemas.Prediction(
            datum=img1,
            annotations=...
     ), …
]


# the user passes a list of all groundtruths and predictions for a list of datums
# this allows us to precompute IOUs at the datum_uid + label_key level
manager.add_data(groundtruths=groundtruths, predictions=predictions)

# the user calls .evaluate() to compute the evaluation
evaluation = manager.evaluate()

# the user must pass all groundtruths and predictions for a given datum at once
# this restriction makes it so we can compute IOUs right away and throw away excess info like rasters, saving a significant amount of memory
with pytest.raises(ValueError):
    manager.add_data_for_datum(groundtruths=groundtruths, predictions=predictions) # throws error since img1 has already been added to the manager's data

# the user must also specify the label map, `convert_annotation_to_type`, etc. when instantiating the object
# once set, these attributes can't be changed since subsequent IOU calculations will become apples-to-oranges with prior calculations
with pytest.raises(ValueError):
    manager.label_map = some_label_map # throws an error since label map can't be changed, only instantiated
```