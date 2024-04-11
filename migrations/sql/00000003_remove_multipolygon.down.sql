alter table annotation add column multipolygon geometry(MultiPolygon);

create index idx_annotation_multipolygon on annotation(multipolygon);
