create extension if not exists "postgis";
create extension if not exists "postgis_raster";

create table label
(
    id         serial
        primary key,
    key        varchar   not null,
    value      varchar   not null,
    created_at timestamp not null,
    unique (key, value)
);

alter table label
    owner to postgres;

create index ix_label_id
    on label (id);

create table model
(
    id         serial
        primary key,
    name       varchar   not null,
    meta       jsonb,
    geo        geography(Geometry, 4326),
    status     varchar   not null,
    created_at timestamp not null
);

alter table model
    owner to postgres;

create index idx_model_geo
    on model using gist (geo);

create unique index ix_model_name
    on model (name);

create index ix_model_id
    on model (id);

create table dataset
(
    id         serial
        primary key,
    name       varchar   not null,
    meta       jsonb,
    geo        geography(Geometry, 4326),
    status     varchar   not null,
    created_at timestamp not null
);

alter table dataset
    owner to postgres;

create unique index ix_dataset_name
    on dataset (name);

create index ix_dataset_id
    on dataset (id);

create index idx_dataset_geo
    on dataset using gist (geo);

create table datum
(
    id         serial
        primary key,
    dataset_id integer   not null
        references dataset,
    uid        varchar   not null,
    meta       jsonb,
    geo        geography(Geometry, 4326),
    created_at timestamp not null,
    unique (dataset_id, uid)
);

alter table datum
    owner to postgres;

create index ix_datum_id
    on datum (id);

create index idx_datum_geo
    on datum using gist (geo);

create table evaluation
(
    id         serial
        primary key,
    dataset_id integer   not null
        references dataset,
    model_id   integer   not null
        references model,
    task_type  varchar   not null,
    settings   jsonb,
    geo        geography(Geometry, 4326),
    status     varchar   not null,
    created_at timestamp not null,
    unique (dataset_id, model_id, task_type, settings)
);

alter table evaluation
    owner to postgres;

create index idx_evaluation_geo
    on evaluation using gist (geo);

create index ix_evaluation_id
    on evaluation (id);

create table annotation
(
    id           serial
        primary key,
    datum_id     integer   not null
        references datum,
    model_id     integer
        references model,
    task_type    varchar   not null,
    meta         jsonb,
    geo          geography(Geometry, 4326),
    created_at   timestamp not null,
    box          geometry(Polygon),
    polygon      geometry(Polygon),
    multipolygon geometry(MultiPolygon),
    raster       raster
);

alter table annotation
    owner to postgres;

create index ix_annotation_id
    on annotation (id);

create index idx_annotation_multipolygon
    on annotation using gist (multipolygon);

create index idx_annotation_polygon
    on annotation using gist (polygon);

create index idx_annotation_raster
    on annotation using gist (st_convexhull(raster));

create index idx_annotation_box
    on annotation using gist (box);

create index idx_annotation_geo
    on annotation using gist (geo);

create table metric
(
    id            serial
        primary key,
    evaluation_id integer   not null
        references evaluation,
    label_id      integer
        references label,
    type          varchar   not null,
    value         double precision,
    parameters    jsonb,
    created_at    timestamp not null
);

alter table metric
    owner to postgres;

create index ix_metric_id
    on metric (id);

create table confusion_matrix
(
    id            serial
        primary key,
    evaluation_id integer   not null
        references evaluation,
    label_key     varchar   not null,
    value         jsonb,
    created_at    timestamp not null
);

alter table confusion_matrix
    owner to postgres;

create index ix_confusion_matrix_id
    on confusion_matrix (id);

create table groundtruth
(
    id            serial
        primary key,
    annotation_id integer
        references annotation,
    label_id      integer   not null
        references label,
    created_at    timestamp not null
);

alter table groundtruth
    owner to postgres;

create index ix_groundtruth_id
    on groundtruth (id);

create table prediction
(
    id            serial
        primary key,
    annotation_id integer
        references annotation,
    label_id      integer   not null
        references label,
    score         double precision,
    created_at    timestamp not null
);

alter table prediction
    owner to postgres;

create index ix_prediction_id
    on prediction (id);
