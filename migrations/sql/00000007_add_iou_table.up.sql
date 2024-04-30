create table iou
(
    id serial primary key,
    groundtruth_annotation_id integer not null references annotation,
    prediction_annotation_id integer not null references annotation,
    value float not null,
    created_at    timestamp not null,
    unique (groundtruth_annotation_id, prediction_annotation_id)
);
