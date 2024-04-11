drop index idx_annotation_multipolygon;

alter table annotation remove column multipolygon;
