# Semantic segmentation

Pass two equally shaped, nonempty 2D integer NumPy arrays and a shared list of
unique label names. Every pixel value indexes that list. Zero is an ordinary
class: its name is `labels[0]`, and its metrics are calculated like every other
class. There are no unlabelled or ignored input pixels.

```python
import numpy as np
from valor_lite.semantic_segmentation import Loader, MetricType, Segmentation

segmentation = Segmentation(
    uid="image-1",
    labels=["sky", "road", "car"],
    groundtruths=np.array([[0, 1], [2, 2]], dtype=np.uint16),
    predictions=np.array([[0, 1], [1, 2]], dtype=np.uint16),
)

loader = Loader.in_memory()
loader.add_data([segmentation])
evaluator = loader.finalize()
metrics = evaluator.compute_precision_recall_iou()
print(metrics[MetricType.Precision][0])
```

Arrays are checked before conversion to `uint16`. A label list can contain up
to 65,536 entries, with indices from 0 through 65,535. Boolean and floating-point
arrays, negative indices, indices outside the label list, and mismatched shapes
are rejected. The annotation derives `shape` and `size` from its arrays.
Validation and loading do not modify caller-owned pixels; arrays already stored
as `uint16` may share memory with the annotation. Do not modify an annotation's
arrays or label mapping between validation and loading.

Each image may have a different label list or ordering. The loader reconciles
classes by name across images and batches. All declared classes are retained,
including classes absent from the pixels. Counts and internal class indices use
64-bit integers. Ingestion stores observed pairs plus one zero-count row for
each declared class absent from both maps. These rows preserve the vocabulary
without changing pixel counts. Full confusion-matrix reporting still requires
space quadratic in the combined number of classes.

Use `metadata` for image-level fields. Optional `groundtruth_metadata` and
`prediction_metadata` dictionaries map local label indices to class metadata:

```python
segmentation = Segmentation(
    uid="image-1",
    labels=["sky", "road", "car"],
    groundtruths=np.array([[0, 1], [2, 2]]),
    predictions=np.array([[0, 1], [1, 2]]),
    metadata={"split": "validation"},
    groundtruth_metadata={2: {"gt_quality": "reviewed"}},
    prediction_metadata={2: {"pd_source": "model-v2"}},
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
mask each side independently: if one side is excluded, the remaining side still
contributes a false positive or false negative. A pixel pair excluded on both
sides is discarded entirely from all counts and metrics.

Retained rows keep their original names and IDs. The boolean `gt_valid` and
`pd_valid` columns record which sides remain active; masked labels never become
class zero. Further filters cannot reactivate an excluded side. The vocabulary
comes from retained rows, including masked-side labels and zero-count rows.
Reloading produces the same vocabulary, masks, and metrics without a separate
label file. Legacy caches without validity columns default to unfiltered sides.
If no rows remain, `filter` raises `EmptyCacheError`.

## Migration from bitmasks

`Segmentation` replaces `list[Bitmask]` with arrays and requires `labels`.
The semantic-segmentation `Bitmask` export and the `shape` and `size` constructor
arguments are removed. To migrate, assign a class index to every pixel and
move mask metadata into the optional dictionaries keyed by that index. An
all-zero array assigns every pixel to `labels[0]`; it does not represent an
empty annotation. Ground truth and prediction arrays are both required.

Formerly uncovered pixels must now receive an explicit class. That class
participates in per-class metrics and mean IoU, which can change results
compared with the old implicit-background behavior.

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
