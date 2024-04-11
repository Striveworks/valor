drop index idx_annotation_polygon;

alter table annotation drop column embedding_id;

drop table embedding cascade;

drop extension if exists "vector";
