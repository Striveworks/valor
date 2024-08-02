create table iou
(
    id serial primary key,
    groundtruth_annotation_id integer not null references annotation,
    prediction_annotation_id integer not null references annotation,
    type varchar not null,
    iou double precision not null,
    unique (groundtruth_annotation_id, prediction_annotation_id, type)
);

create index ix_iou_id
    on iou (id);
