# Semantic segmentation

Pass equally shaped, nonempty 2D integer arrays and a list of unique foreground
label names. **Pixel value 0 is reserved for background.** For N pixel indices
(0 through N-1), supply N-1 names: positive value `i` refers to `labels[i - 1]`.
For example, `labels=["cat", "dog"]` maps 0 to background, 1 to cat, and 2 to dog.
Do not include a background name in `labels`.

Arrays are validated before conversion to `uint16`. Values must be between 0
and `len(labels)`, inclusive; at most 65,535 foreground names are supported.
An empty label list permits only background pixels. Boolean, floating-point,
negative, out-of-range, empty, and mismatched inputs are rejected. `shape` and
`size` are derived from the arrays. Loading does not modify the inputs, though
`uint16` arrays may share storage with the annotation.

Each image may have a different foreground label list or ordering. The loader
reconciles classes by name across images and batches. Declared classes absent
from both maps are retained in zero-count rows. See [metric semantics](metrics.md)
for their effect on mean IoU and for background and filtering behavior.

## Metadata and filtering

Use `metadata` for image fields. `groundtruth_metadata` and `prediction_metadata`
map pixel values to class metadata: keys 1 through `len(labels)` select foreground
classes, and key 0 selects background. Cached metadata combines image, ground
truth, and prediction fields in that order; later fields take precedence.
Evaluator-reserved fields cannot be overwritten. Use separate field names for
ground truth and prediction metadata when filtering each side independently.

Datum filters discard complete rows. Annotation filters remap excluded sides
to background; rows failing both side filters are discarded. Excluded foreground
labels disappear from the vocabulary unless retained on the other side. Explicit
label-name overrides are preserved for retained IDs. If no rows remain, `filter`
raises `EmptyCacheError`.

Persistent caches recover labels from their rows. Keep `metadata.json` and the
`counts` directory together when moving a cache; no separate label file is needed.
Background and excluded sides use cache ID -1 with a null name, which maps to
confusion-matrix index 0. Subsequent filters see these sides as background.
Legacy caches remain readable, including noncontiguous label IDs. Name overrides
can be supplied to `load`.

## Migration from bitmasks

`Segmentation` replaces `list[Bitmask]` with arrays and requires `labels`.
The semantic-segmentation `Bitmask` export and the `shape` and `size` constructor
arguments are removed. Keep uncovered pixels at 0 and assign `labels[i]` to
pixel value `i + 1`. Move mask metadata into the dictionaries keyed by that pixel
value. Ground truth and prediction arrays are both required; an all-zero array
represents background. The existing metric formulas are preserved.

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
