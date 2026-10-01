# Semantic segmentation

Pass two equally shaped, nonempty 2D integer NumPy arrays and a shared list of
unique foreground label names. **Pixel value 0 is reserved for background.**
For N possible pixel values (0 through N-1), supply N-1 names: positive pixel
value `i` refers to `labels[i - 1]`. Do not include a background name in `labels`.
Background matches contribute to pixel accuracy, but background has no
per-class metrics and does not contribute to mean IoU.

```python
import numpy as np
from valor_lite.semantic_segmentation import Loader, MetricType, Segmentation

segmentation = Segmentation(
    uid="image-1",
    labels=["sky", "road", "car"],
    groundtruths=np.array([[0, 1], [2, 3]], dtype=np.uint16),
    predictions=np.array([[0, 1], [1, 3]], dtype=np.uint16),
)

loader = Loader.in_memory()
loader.add_data([segmentation])
evaluator = loader.finalize()
metrics = evaluator.compute_precision_recall_iou()
print(metrics[MetricType.Precision][0])
```

Here, 0 is background, 1 is sky, 2 is road, and 3 is car.

Arrays are checked before conversion to `uint16`. A label list can contain up
to 65,535 foreground names. Pixel values must be between 0 and `len(labels)`,
inclusive. An empty label list permits only background pixels. Boolean and
floating-point arrays, negative or out-of-range values, and mismatched shapes
are rejected. The annotation derives `shape` and `size` from its arrays.
Validation and loading do not modify caller-owned pixels; arrays already stored
as `uint16` may share memory with the annotation. Do not modify an annotation's
arrays or label mapping between validation and loading.

Each image may have a different foreground label list or ordering. Zero always
remains background. The loader reconciles foreground classes by name across
images and batches. All declared foreground classes are retained,
including classes absent from the pixels. Counts and internal class indices use
64-bit integers. Ingestion stores observed pairs plus one zero-count row for
each declared class absent from both maps. These rows preserve the vocabulary
without changing pixel counts. Full confusion-matrix reporting still requires
space quadratic in the combined number of classes.

Use `metadata` for image-level fields. Optional `groundtruth_metadata` and
`prediction_metadata` dictionaries map pixel values to class metadata. Keys
1 through `len(labels)` refer to foreground classes; key 0 refers to background:

```python
segmentation = Segmentation(
    uid="image-1",
    labels=["sky", "road", "car"],
    groundtruths=np.array([[0, 1], [2, 3]]),
    predictions=np.array([[0, 1], [1, 3]]),
    metadata={"split": "validation"},
    groundtruth_metadata={3: {"gt_quality": "reviewed"}},
    prediction_metadata={3: {"pd_source": "model-v2"}},
)
loader = Loader.in_memory(metadata_fields=[
    ("split", "string"), ("gt_quality", "string"), ("pd_source", "string"),
])
loader.add_data([segmentation])
```

Metadata on a cached pair combines image fields, ground truth class fields, and
prediction class fields in that order. Later fields override earlier fields
with the same name; evaluator-reserved fields cannot be overwritten. Use
separate field names for ground truth and prediction metadata when filtering
each side independently.

Persistent caches recover their vocabulary from the label names and IDs in
the cached rows. There is no separate label file. Keep `metadata.json` and the
`counts` directory together when moving a cache. Legacy caches remain readable,
including caches with noncontiguous label IDs.

Datum filtering discards complete rows. Ground truth and prediction filters
apply independently and remap excluded annotations to background. A surviving
foreground annotation contributes a false positive or false negative. If the
other side is already background, the pair becomes a background match and
counts as correct for accuracy. A row failing both side filters is discarded
entirely from counts and metrics.

Background and excluded sides share the existing cache representation: ID
`-1` with a null name, mapped to confusion-matrix row or column 0. There is no
separate exclusion class. Subsequent filters see these sides as background;
the original excluded label cannot be recovered. The vocabulary comes only
from retained foreground annotations, including zero-count rows. Foreground
labels excluded from both sides disappear from per-class metrics and mean IoU.
Labels retained on either side still contribute.
Filtering preserves explicit label-name overrides for retained IDs. Reloading
reconstructs the vocabulary from included label IDs and names in the cached
rows; name overrides can be supplied to `load`. If no rows remain, `filter`
raises `EmptyCacheError`.

## Migration from bitmasks

`Segmentation` replaces `list[Bitmask]` with arrays and requires `labels`.
The semantic-segmentation `Bitmask` export and the `shape` and `size` constructor
arguments are removed. To migrate, assign a class index to every pixel and
move mask metadata into the optional dictionaries keyed by that pixel value.
Keep uncovered pixels at 0 and assign foreground label `labels[i]` to pixel
value `i + 1`. An all-zero array represents entirely background. Ground truth
and prediction arrays are both required.

The existing confusion-matrix convention and metric formulas are preserved:
row and column 0 are background, foreground classes occupy the remaining
positions, and accuracy includes background matches. No background name is
required in the label list.

## API

::: valor_lite.semantic_segmentation.Segmentation
    options:
        show_root_heading: true
        show_source: true

::: valor_lite.semantic_segmentation.Loader
    options:
        show_root_heading: true
        show_source: true

::: valor_lite.semantic_segmentation.Evaluator
    options:
        show_root_heading: true
        show_source: true
