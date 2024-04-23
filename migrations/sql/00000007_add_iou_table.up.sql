create table iou
(
    groundtruth_id integer not null references groundtruth,
    prediction_id integer not null references prediction,
    value float not null,
    created_at    timestamp not null,
    unique (groundtruth_id, prediction_id)
);
