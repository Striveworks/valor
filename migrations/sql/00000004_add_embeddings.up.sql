create extension if not exists "vector";

create table embedding
(
    id         serial primary key,
    value      vector not null,
    created_at timestamp not null
);

create index ix_embedding_id
    on embedding (id);

ALTER TABLE annotation ADD embedding_id integer references embedding;
